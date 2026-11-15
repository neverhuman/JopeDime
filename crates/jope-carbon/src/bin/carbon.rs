//! Multi-role `carbon` CLI and AWS Lambda custom-runtime entrypoint (`version`, `self-verify`,
//! `doctor`, `quickstart`, `register`, `login`, `whoami`, `logout`, `ping`, `run`, `bench`,
//! `redteam`, `token-mint`, `plan`, `spot-check`, plus feature-gated operator/server subcommands).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::IsTerminal as _;
#[cfg(feature = "serve")]
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(feature = "engine")]
use std::sync::Arc;

use base64::Engine as _;
use clap::{ArgAction, Parser, Subcommand};
use jope_api::{
    carbon::{CarbonExplainResponse, CarbonExportLangWire, CarbonTierWire},
    Effort,
};
use jope_carbon::observability::{
    emit_cli_error, init_cli_observability, install_sigint_flag, write_stdout_line,
    CarbonColorMode, CarbonLogFormat, EXIT_AUTH_EXPIRED,
};
#[cfg(any(
    feature = "engine",
    feature = "eval",
    feature = "serve",
    feature = "emu"
))]
use jope_carbon::CarbonRole;
use jope_carbon::{
    build_client_for_handle, decode_p256_pubkey_b64, estimate_ms_per_candidate_1vcpu,
    evaluate_spot_hybrid_gates, execute_carbon_cli_benchmark_with_token,
    execute_carbon_cli_run_with_ux, execute_carbon_redteam_suite, execute_predict_csv_with_handle,
    execute_scientific_benchmark_suite, execute_self_upgrade_with_options,
    generate_nonlinear_benchmark_rows, inspect_tabular_file_with_tier, list_saved_model_handles,
    list_stored_cloud_tokens, load_delimited_table, load_model_handle_by_spec,
    load_stored_cloud_token, now_epoch_ms, parse_flexible_numeric_f64,
    persist_model_handle_with_options, plan_carbon_execution, query_handle_status,
    remove_stored_cloud_token, render_benchmark_markdown, render_pretty_run_scorecard,
    render_redteam_markdown, render_scientific_benchmark_markdown, resolve_objective_argument,
    resolve_public_cloud_endpoint, save_stored_cloud_token, sha256_lower_hex,
    shred_handle_and_local_keys, verified_absolute_jopedime_config_dir,
    verify_ecdsa_p256_spki_pem_signature, CarbonAuthVerifier, CarbonCapabilityClaims,
    CarbonCliRunOptions, CarbonCliRunUxConfig, CarbonEffortTier, CarbonPlanInput,
    CarbonProbedEndpoint, CarbonProductTier, CarbonPublicCloudsManifest,
    CarbonScientificBenchmarkOptions, CarbonTabularInspectReport, CarbonThinClient,
    MonthlyWindowMetrics, SpotAdvisorThresholds, CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET,
    CARBON_GIT_SHA, CARBON_PKG_VERSION, CARBON_RELEASE_TAG, DEFAULT_PUBLIC_CLOUDS_URL,
    FALLBACK_PUBLIC_CLOUDS_URL, MAX_MINT_TOKEN_TTL_SECS,
};
#[cfg(feature = "engine")]
use jope_carbon::{
    execute_carbon_deploy, execute_carbon_package, execute_carbon_rollback_multi,
    generate_public_clouds_manifest, mint_capability_token_from_vault, open_eval_invoker,
    open_remote_cas, resolve_cell_stack_endpoint, resolve_engine_secrets,
    run_lambda_custom_runtime_loop, CarbonDeployOptions, CarbonEngineWorker, CarbonEvalWorker,
    CarbonLiveAliasVersions, CarbonPackageOptions,
};
#[cfg(feature = "emu")]
use jope_carbon::{
    execute_carbon_spec_report, execute_carbon_spec_verify, CarbonEmuConfig,
    CarbonLocalLambdaEmulator, CarbonSpecReportOptions, CarbonSpecVerifyOptions,
};
#[cfg(feature = "serve")]
use jope_carbon::{CarbonEnterpriseServeConfig, CarbonEnterpriseServer};
use jope_prep::ObjectiveSpec;

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

const EMBEDDED_RELEASE_SIGNING_PUB_PEM: &str = include_str!("../../../../docs/release/signing.pub");
const EXPECTED_SIGNING_PUB_SHA256: &str =
    "de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e";

fn canonical_embedded_release_signing_pub_pem() -> String {
    EMBEDDED_RELEASE_SIGNING_PUB_PEM.replace("\r\n", "\n")
}

#[derive(Debug, Parser)]
#[command(
    name = "carbon",
    version = LONG_VERSION,
    args_conflicts_with_subcommands = true,
    about = "Carbon zero-knowledge serverless CLI client, benchmark & red-team harness, Lambda runtime, Enterprise server, and CI emulator"
)]
struct Cli {
    /// Increase diagnostic verbosity on stderr (`-v` = debug, `-vv` = trace; overrides `CARBON_LOG`).
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress interactive progress and diagnostic messages on stderr (`-q` / `--quiet`).
    #[arg(short = 'q', long = "quiet", global = true)]
    quiet: bool,

    /// Diagnostic output format on stderr (`text` or `json`).
    #[arg(long = "log-format", default_value = "text", global = true)]
    log_format: String,

    /// ANSI color policy (`auto`, `always`, or `never`; respects `NO_COLOR`).
    #[arg(long = "color", default_value = "auto", global = true)]
    color: String,

    /// Positional shorthand (`carbon data.csv` / `data.csv.gz` / `data.jsonl` / `-`): conditions & encrypts
    /// client-side, runs a `fast` zero-knowledge cloud job, decrypts predictions, and shreds remote blobs.
    #[arg(value_name = "CSV_FILE")]
    csv_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum FitnessCommand {
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

#[derive(Debug, clap::Args)]
struct CommonTrainArgs {
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
    /// Optional explicit tenant ID (automatically extracted from `ctk1.*` token claims when omitted).
    #[arg(long)]
    tenant: Option<String>,
    /// Select Free anonymous tier (`<= 5,000` rows, `fast`/`medium`).
    #[arg(long, conflicts_with_all = ["pro", "tier"])]
    free: bool,
    /// Select authenticated Pro tier (`<= 100,000` rows, `fast`..`ultra` multi-wave + `HybridFanOut`).
    #[arg(long, conflicts_with_all = ["free", "tier"])]
    pro: bool,
    /// Explicit tier (`free`, `pro`, or `enterprise`).
    #[arg(long, conflicts_with_all = ["free", "pro"])]
    tier: Option<String>,
    /// Signed `ctk1.*` capability token for Pro/Enterprise runs.
    #[arg(long, conflicts_with = "token_file")]
    token: Option<String>,
    /// File containing a signed `ctk1.*` capability token (`-` reads stdin).
    #[arg(long, conflicts_with = "token")]
    token_file: Option<PathBuf>,
    #[arg(long, default_value = "fast")]
    effort: String,
    /// Regression objective preset (`rmse`, `mae`, `pinball(0.9)`), inline JSON, or `@file.json` (`--fitness` alias).
    #[arg(long, visible_alias = "fitness", default_value = "rmse")]
    objective: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Input CSV has no header row (columns are auto-named `f0..fk`; `--target -1` selects final column).
    #[arg(long)]
    no_header: bool,
    /// Override delimiter character (`,`, `;`, `tab` / `\t`, or `|`; auto-detected when omitted).
    #[arg(long)]
    delimiter: Option<String>,
    /// Holdout split strategy (`random` deterministic hash split, or `forward` chronological tail split).
    #[arg(long)]
    split: Option<String>,
    /// Fold-safe post-hoc prediction calibration (`auto`, `none`, `shift`, or `affine`).
    #[arg(long)]
    calibration: Option<String>,
    /// Additional comma-separated custom missing-value tokens (`--na-values -999,MISSING`).
    #[arg(long, value_delimiter = ',')]
    na_values: Vec<String>,
    /// Coerce `Inf` / `-Inf` cells to missing (`NaN`) instead of failing closed.
    #[arg(long)]
    inf_as_nan: bool,
    /// Embed the active `ctk1.*` capability token inside an explicit `--save` file (default strips token outside `$CARBON_CONFIG_DIR/runs/`).
    #[arg(long)]
    embed_token: bool,
    /// Validate conditioning, objective, admission limits, and execution cost locally with zero network I/O.
    #[arg(long)]
    dry_run: bool,
    /// Permit unencrypted dataset upload to a non-loopback self-hosted `carbon serve` endpoint without `--enable-user-envelope`.
    #[arg(long)]
    allow_unencrypted_enterprise: bool,
    /// Force pure JSON output on stdout (`--format json` shorthand).
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
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
    /// followed by mandatory `/ping`, `/version`, and 16-vector live `carbon redteam` verification.
    #[cfg(feature = "engine")]
    Deploy {
        /// Path to `release-manifest.json` produced by `carbon package`.
        #[arg(long)]
        manifest: PathBuf,
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
        /// Validate template & preview CloudFormation ChangeSet resource diff without executing.
        #[arg(long)]
        dry_run: bool,
        /// Skip post-deploy 16-vector red-team suite (not recommended outside offline diagnostics).
        #[arg(long)]
        skip_redteam: bool,
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        /// Path to a file containing the `ctk1.*` capability token (`-` reads stdin).
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        /// Optional file containing a `ctk1.*` capability token to inspect.
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
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
        #[arg(long, default_value = "json")]
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Cryptographically shred remote S3/CAS objects (`DELETE /v1/carbon/runs/{id}`) AND zero-overwrite local key handles.
    Shred {
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        /// Output format on stdout (`auto`, `pretty`, or `json`).
        #[arg(long, default_value = "auto")]
        format: String,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Export the winning symbolic GP champion to standalone zero-dependency predictor source code (`POST /v1/carbon/runs/{id}/export`).
    Export {
        /// Path to a saved `0600` model handle (`--save model.carbon.json` from `carbon fit`).
        #[arg(long, short = 'm', conflicts_with = "run")]
        model: Option<PathBuf>,
        /// Saved `run_id` (or `latest`) from `$HOME/.config/jopedime/runs/` when `--model` is omitted.
        #[arg(long, visible_alias = "run-id", conflicts_with = "model")]
        run: Option<String>,
        /// Target standalone language (`python`, `rust`, `rust-refittable`, `js`, `c`, or `cpp`).
        #[arg(long, default_value = "python")]
        lang: String,
        /// Optional output file path (or `-` to write source code to stdout).
        #[arg(long, short = 'o')]
        out: Option<PathBuf>,
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        /// Emit structured JSON receipt on stdout instead of raw source text.
        #[arg(long)]
        json: bool,
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
        /// Emit pure JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Check or install ECDSA-P256 cryptographically signed Carbon CLI updates (`SHA256SUMS.txt.sig`).
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
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        /// File containing a signed `ctk1.*` Pro capability token (`-` reads stdin).
        #[arg(long, conflicts_with = "token")]
        token_file: Option<PathBuf>,
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
    /// Mint a signed `ctk1.*` capability token using AWS Secrets Manager (`--env`) or `CARBON_AUTH_SECRET_HEX`.
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
    /// Boot the local multi-cell Lambda Runtime API + S3 CAS emulator (`--self-test` for CI).
    #[cfg(feature = "emu")]
    Emu {
        #[arg(long)]
        work_dir: PathBuf,
        #[arg(long)]
        self_test: bool,
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
        /// Named public Carbon cloud (`prod` or `dev`, defaults to `prod`).
        #[arg(long, default_value = "prod")]
        cloud: String,
        /// Optional path or HTTPS URL overriding the public `carbon-clouds.json` manifest.
        #[arg(long)]
        clouds_manifest: Option<String>,
        /// Optional explicit endpoint URL(s) overriding `--cloud`.
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Search effort (`fast` or `medium` on Free; up to `ultra` when logged into Pro).
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

fn parse_product_tier(raw: &str) -> Result<CarbonProductTier, String> {
    match raw.trim() {
        "free" => Ok(CarbonProductTier::Free),
        "pro" => Ok(CarbonProductTier::Pro),
        "enterprise" => Ok(CarbonProductTier::no_limit()),
        other => Err(format!(
            "unknown product tier `{other}` (expected free, pro, or enterprise)"
        )),
    }
}

fn parse_tier_wire(raw: &str) -> Result<CarbonTierWire, String> {
    match raw.trim() {
        "free" => Ok(CarbonTierWire::Free),
        "pro" => Ok(CarbonTierWire::Pro),
        "enterprise" => Ok(CarbonTierWire::Enterprise),
        other => Err(format!(
            "unknown tier `{other}` (expected free, pro, or enterprise)"
        )),
    }
}

fn parse_api_effort(raw: &str) -> Result<Effort, String> {
    match raw.trim() {
        "fast" => Ok(Effort::Fast),
        "medium" => Ok(Effort::Medium),
        "high" => Ok(Effort::High),
        "ultra" => Ok(Effort::Ultra),
        other => Err(format!(
            "unknown effort `{other}` (expected fast, medium, high, or ultra)"
        )),
    }
}

#[cfg(feature = "serve")]
fn decode_required_32byte_hex(env_key: &str) -> Result<[u8; 32], String> {
    let val = env::var(env_key).map_err(|_| format!("missing required {env_key}"))?;
    let raw = hex::decode(val.trim()).map_err(|_| format!("{env_key} must be 64-char hex"))?;
    if raw.len() != 32 {
        return Err(format!("{env_key} must decode to 32 bytes"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

#[cfg(feature = "serve")]
fn decode_or_ephemeral_32byte_secret(
    env_key: &str,
    allow_ephemeral: bool,
) -> Result<[u8; 32], String> {
    match env::var(env_key) {
        Ok(val) if !val.trim().is_empty() => decode_required_32byte_hex(env_key),
        _ if allow_ephemeral => {
            let mut out = [0u8; 32];
            aws_lc_rs::rand::fill(&mut out)
                .map_err(|_| format!("CSPRNG failed generating ephemeral {env_key}"))?;
            Ok(out)
        }
        _ => Err(format!(
            "non-loopback `carbon serve` requires {env_key} (64-char hex) or explicit --dev-ephemeral"
        )),
    }
}

fn resolve_stack_endpoint_if_compiled(env_name: &str, region: &str) -> Result<String, String> {
    #[cfg(feature = "engine")]
    {
        resolve_cell_stack_endpoint(env_name, region)
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (env_name, region);
        Err(
            "--env CloudFormation stack resolution requires the operator build (--features engine); public clients use --cloud <prod|dev> or --endpoint <URL>"
                .to_owned(),
        )
    }
}

fn mint_vault_token_if_compiled(
    env_name: &str,
    region: &str,
    tenant: &str,
    tier: CarbonTierWire,
    ttl_secs: u64,
    admin: bool,
) -> Result<String, String> {
    #[cfg(feature = "engine")]
    {
        mint_capability_token_from_vault(env_name, region, tenant, tier, ttl_secs, admin)
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (env_name, region, tenant, tier, ttl_secs, admin);
        Err(
            "--env Secrets Manager token minting requires the operator build (--features engine); pass CARBON_AUTH_SECRET_HEX or --token <ctk1...>"
                .to_owned(),
        )
    }
}

fn compiled_binary_roles() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut roles = vec!["cli"];
    #[cfg(feature = "engine")]
    roles.push(CarbonRole::EngineOrchestrate.as_str());
    #[cfg(feature = "eval")]
    roles.push(CarbonRole::EvalBatch.as_str());
    #[cfg(feature = "serve")]
    roles.push(CarbonRole::EnterpriseServe.as_str());
    #[cfg(feature = "emu")]
    roles.push(CarbonRole::LocalEmulator.as_str());
    roles
}

fn read_cli_token_source(
    inline_token: Option<String>,
    token_file: Option<PathBuf>,
) -> Result<Option<String>, String> {
    if let Some(tok) = inline_token {
        let trimmed = tok.trim().to_owned();
        if trimmed.is_empty() {
            return Err("--token must not be empty".to_owned());
        }
        jope_carbon::observability::log_warn(
            "passing --token on the command line exposes credentials to local process listings; prefer --token-file <PATH> (or '-') or CARBON_AUTH_TOKEN",
        );
        return Ok(Some(trimmed));
    }
    if let Some(path) = token_file {
        let raw = if path.as_os_str() == "-" {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("read token from stdin: {e}"))?;
            s
        } else {
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?
        };
        let trimmed = raw.trim().to_owned();
        if trimmed.is_empty() {
            return Err(format!("token file {} is empty", path.display()));
        }
        return Ok(Some(trimmed));
    }
    Ok(None)
}

fn resolve_explicit_tier_flag(
    free: bool,
    pro: bool,
    tier: Option<&str>,
) -> Result<Option<CarbonTierWire>, String> {
    if free {
        Ok(Some(CarbonTierWire::Free))
    } else if pro {
        Ok(Some(CarbonTierWire::Pro))
    } else if let Some(t) = tier {
        parse_tier_wire(t).map(Some)
    } else {
        Ok(None)
    }
}

struct ResolvedCliTarget {
    endpoints: Vec<String>,
    pinned_keks: BTreeMap<String, String>,
    probed_client: CarbonThinClient,
    first_probe: CarbonProbedEndpoint,
    tier: CarbonTierWire,
    tenant_id: String,
    auth_token: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn resolve_cli_target_and_auth(
    mut endpoints: Vec<String>,
    mut cloud: Option<String>,
    clouds_manifest: Option<&str>,
    environment: Option<&str>,
    region: &str,
    explicit_tier: Option<CarbonTierWire>,
    explicit_tenant: Option<String>,
    default_tenant: &str,
    inline_token: Option<String>,
    token_file: Option<PathBuf>,
    default_tier_when_self_hosted: CarbonTierWire,
) -> Result<ResolvedCliTarget, (ExitCode, String)> {
    if endpoints.is_empty() && cloud.is_none() && environment.is_none() {
        if let Ok(env_ep) = env::var("CARBON_ENDPOINT") {
            let trimmed = env_ep.trim();
            if !trimmed.is_empty() {
                endpoints.push(trimmed.to_owned());
            }
        }
        if endpoints.is_empty() {
            cloud = Some("prod".to_owned());
        }
    }

    let mut pinned_keks: BTreeMap<String, String> = BTreeMap::new();
    if let Some(cloud_name) = &cloud {
        if endpoints.is_empty() || clouds_manifest.is_some() {
            let entry = resolve_public_cloud_endpoint(cloud_name, clouds_manifest)
                .map_err(|e| (ExitCode::from(2), e))?;
            pinned_keks = entry.pinned_keks_by_endpoint();
            if endpoints.is_empty() {
                endpoints = entry.candidate_endpoints();
            }
        }
    }
    if let Some(env_name) = environment {
        if endpoints.is_empty() {
            let url = resolve_stack_endpoint_if_compiled(env_name, region)
                .map_err(|e| (ExitCode::from(2), e))?;
            endpoints.push(url);
        }
    }
    if endpoints.is_empty() {
        return Err((
            ExitCode::from(2),
            "pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>".to_owned(),
        ));
    }

    let (probed_client, probed) =
        CarbonThinClient::connect_fastest_pinned(endpoints.clone(), &pinned_keks)
            .map_err(|e| (ExitCode::from(1), e))?;
    let first_probe = probed
        .first()
        .cloned()
        .ok_or_else(|| (ExitCode::from(1), "empty probe response".to_owned()))?;
    endpoints = probed.iter().map(|p| p.base_url.clone()).collect();

    let now_secs = now_epoch_ms() / 1_000;
    let cli_tok =
        read_cli_token_source(inline_token, token_file).map_err(|e| (ExitCode::from(2), e))?;
    let env_tok = env::var("CARBON_AUTH_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    let has_local_hmac_secret = env::var("CARBON_AUTH_SECRET_HEX")
        .ok()
        .is_some_and(|s| !s.trim().is_empty());

    let mut active_token: Option<String> = if explicit_tier == Some(CarbonTierWire::Free) {
        cli_tok
    } else {
        cli_tok.or(env_tok)
    };
    if active_token.is_none() {
        if let Some(lookup_cloud) = cloud.as_deref() {
            if let Ok(Some(stored)) = load_stored_cloud_token(lookup_cloud) {
                if explicit_tier == Some(CarbonTierWire::Free) {
                    if stored.tier == CarbonTierWire::Free && stored.exp_epoch_secs > now_secs {
                        active_token = Some(stored.capability_token);
                    }
                } else if stored.exp_epoch_secs > now_secs {
                    active_token = Some(stored.capability_token);
                } else if matches!(
                    explicit_tier,
                    Some(CarbonTierWire::Pro | CarbonTierWire::Enterprise)
                ) {
                    return Err((
                        ExitCode::from(2),
                        format!(
                            "stored capability token for cloud '{lookup_cloud}' expired at epoch {}; run 'carbon login --cloud {lookup_cloud} --token <ctk1...>' or pass --token",
                            stored.exp_epoch_secs
                        ),
                    ));
                }
            }
        }
    }

    let mut effective_tenant = explicit_tenant
        .clone()
        .unwrap_or_else(|| default_tenant.to_owned());
    let mut effective_tier = explicit_tier.unwrap_or({
        if first_probe.enterprise_self_hosted || has_local_hmac_secret || environment.is_some() {
            default_tier_when_self_hosted
        } else {
            CarbonTierWire::Free
        }
    });

    if let Some(tok) = &active_token {
        let claims = CarbonCapabilityClaims::decode_unverified(tok)
            .map_err(|e| (ExitCode::from(2), e.to_string()))?;
        if now_secs > claims.exp_epoch_secs {
            return Err((
                ExitCode::from(2),
                format!(
                    "capability token for tenant '{}' has expired (exp={})",
                    claims.tenant_id, claims.exp_epoch_secs
                ),
            ));
        }
        if let Some(exp_tenant) = &explicit_tenant {
            if exp_tenant.trim() != claims.tenant_id {
                return Err((
                    ExitCode::from(2),
                    format!(
                        "--tenant '{}' does not match token tenant_id '{}' (omit --tenant to auto-use the token's tenant_id)",
                        exp_tenant.trim(),
                        claims.tenant_id
                    ),
                ));
            }
        }
        effective_tenant = claims.tenant_id;
        if let Some(req_tier) = explicit_tier {
            if req_tier == CarbonTierWire::Pro && claims.tier == CarbonTierWire::Free {
                return Err((
                    ExitCode::from(2),
                    "flag --pro requires a Pro or Enterprise capability token, but provided token has tier 'free'".to_owned(),
                ));
            }
            effective_tier = req_tier;
        } else {
            effective_tier = claims.tier;
        }
    } else if let Some(env_name) = environment {
        if effective_tier != CarbonTierWire::Free && !has_local_hmac_secret {
            let minted = mint_vault_token_if_compiled(
                env_name,
                region,
                &effective_tenant,
                effective_tier,
                3_600,
                false,
            )
            .map_err(|e| (ExitCode::from(2), e))?;
            active_token = Some(minted);
        }
    } else if !first_probe.enterprise_self_hosted
        && !has_local_hmac_secret
        && matches!(
            effective_tier,
            CarbonTierWire::Pro | CarbonTierWire::Enterprise
        )
    {
        let cloud_hint = cloud.as_deref().unwrap_or("prod");
        return Err((
            ExitCode::from(2),
            format!(
                "Pro/Enterprise tier on public cloud '{cloud_hint}' requires a capability token; pass --token <ctk1...> / --token-file <PATH> or run 'carbon login --cloud {cloud_hint} --token <ctk1...>'"
            ),
        ));
    }

    Ok(ResolvedCliTarget {
        endpoints,
        pinned_keks,
        probed_client,
        first_probe,
        tier: effective_tier,
        tenant_id: effective_tenant,
        auth_token: active_token,
    })
}

fn parse_sha256sums_entries(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let Some(hex_part) = parts.next() else {
            continue;
        };
        let rest = parts.next().unwrap_or("").trim().trim_start_matches('*');
        if hex_part.len() != 64
            || !hex_part.chars().all(|c| c.is_ascii_hexdigit())
            || rest.is_empty()
        {
            return Err(format!(
                "malformed SHA256SUMS.txt line {}: '{line}'",
                idx + 1
            ));
        }
        out.push((hex_part.to_ascii_lowercase(), rest.to_owned()));
    }
    if out.is_empty() {
        return Err("SHA256SUMS.txt contains zero digest entries".to_owned());
    }
    Ok(out)
}

fn execute_self_verify_command(
    manifest_arg: Option<PathBuf>,
    signature_arg: Option<PathBuf>,
    clouds_arg: Option<PathBuf>,
    allow_test_pubkey: Option<PathBuf>,
    require_manifest: bool,
    binary_only: bool,
) -> ExitCode {
    let exe_path = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: locate current executable: {e}");
            return ExitCode::from(1);
        }
    };
    let exe_bytes = match fs::read(&exe_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: read executable {}: {e}", exe_path.display());
            return ExitCode::from(1);
        }
    };
    let exe_sha256 = sha256_lower_hex(&exe_bytes);

    let canonical_embedded_pub_pem = canonical_embedded_release_signing_pub_pem();
    let embedded_pub_sha = sha256_lower_hex(canonical_embedded_pub_pem.as_bytes());
    if embedded_pub_sha != EXPECTED_SIGNING_PUB_SHA256 {
        eprintln!(
            "error: embedded release public key SHA-256 mismatch (got {embedded_pub_sha}, expected {EXPECTED_SIGNING_PUB_SHA256})"
        );
        return ExitCode::from(1);
    }

    let (pubkey_pem, using_test_pubkey) = if let Some(test_pub) = &allow_test_pubkey {
        match fs::read_to_string(test_pub) {
            Ok(pem) => (pem.replace("\r\n", "\n"), true),
            Err(e) => {
                eprintln!(
                    "error: read --allow-test-pubkey {}: {e}",
                    test_pub.display()
                );
                return ExitCode::from(2);
            }
        }
    } else {
        (canonical_embedded_pub_pem, false)
    };

    let config_dir_opt = verified_absolute_jopedime_config_dir().ok();
    let effective_clouds_path = clouds_arg.or_else(|| {
        config_dir_opt
            .as_ref()
            .map(|d| d.join("carbon-clouds.json"))
            .filter(|p| p.is_file())
    });

    let mut clouds_verified = false;
    let mut clouds_sha256: Option<String> = None;
    let mut kek_pins_verified = 0usize;
    if let Some(cpath) = &effective_clouds_path {
        let raw_clouds = match fs::read(cpath) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read clouds manifest {}: {e}", cpath.display());
                return ExitCode::from(1);
            }
        };
        let parsed: CarbonPublicCloudsManifest = match serde_json::from_slice(&raw_clouds) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: invalid JSON in {}: {e}", cpath.display());
                return ExitCode::from(1);
            }
        };
        for (cname, cell) in &parsed.clouds {
            let pins = cell.pinned_keks_by_endpoint();
            if pins.is_empty() {
                eprintln!(
                    "error: cloud '{cname}' in {} has zero P-256 KEK pins",
                    cpath.display()
                );
                return ExitCode::from(1);
            }
            for (ep, pin_b64) in &pins {
                if let Err(e) = decode_p256_pubkey_b64(pin_b64) {
                    eprintln!("error: invalid P-256 KEK pin for '{cname}' endpoint '{ep}': {e}");
                    return ExitCode::from(1);
                }
                kek_pins_verified += 1;
            }
        }
        clouds_sha256 = Some(sha256_lower_hex(&raw_clouds));
        clouds_verified = true;
    }

    let explicit_manifest = manifest_arg.is_some();
    let config_sig_exists = config_dir_opt
        .as_ref()
        .is_some_and(|d| d.join("SHA256SUMS.txt.sig").is_file());
    let must_verify_manifest = !binary_only
        && (explicit_manifest
            || require_manifest
            || signature_arg.is_some()
            || effective_clouds_path.is_some()
            || config_sig_exists);
    let effective_manifest_path = if binary_only {
        None
    } else {
        manifest_arg.or_else(|| {
            config_dir_opt
                .as_ref()
                .map(|d| d.join("SHA256SUMS.txt"))
                .filter(|p| p.is_file())
        })
    };

    let mut manifest_verified = false;
    let mut matched_binary_asset: Option<String> = None;
    let mut manifest_sha256: Option<String> = None;
    if let Some(mpath) = &effective_manifest_path {
        let sig_path = signature_arg.unwrap_or_else(|| {
            let mut s = mpath.as_os_str().to_os_string();
            s.push(".sig");
            PathBuf::from(s)
        });
        let manifest_bytes = match fs::read(mpath) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read {}: {e}", mpath.display());
                return ExitCode::from(1);
            }
        };
        let sig_raw = match fs::read_to_string(&sig_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: read {}: {e}", sig_path.display());
                return ExitCode::from(1);
            }
        };
        let clean_b64: String = sig_raw.chars().filter(|c| !c.is_whitespace()).collect();
        let sig_der = match base64::engine::general_purpose::STANDARD.decode(clean_b64.as_bytes()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "error: malformed base64 signature {}: {e}",
                    sig_path.display()
                );
                return ExitCode::from(1);
            }
        };
        if let Err(err) =
            verify_ecdsa_p256_spki_pem_signature(&pubkey_pem, &manifest_bytes, &sig_der)
        {
            eprintln!(
                "error: {} failed ECDSA-P256 verification against {}: {err}",
                mpath.display(),
                if using_test_pubkey {
                    "test public key"
                } else {
                    "embedded owner release public key"
                }
            );
            return ExitCode::from(1);
        }
        let manifest_text = match std::str::from_utf8(&manifest_bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {} is not valid UTF-8: {e}", mpath.display());
                return ExitCode::from(1);
            }
        };
        let entries = match parse_sha256sums_entries(manifest_text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        };
        for (digest, name) in &entries {
            if digest == &exe_sha256 {
                matched_binary_asset = Some(name.clone());
                break;
            }
        }
        if matched_binary_asset.is_none() {
            eprintln!(
                "error: running executable SHA-256 ({exe_sha256}) does not match any entry in signed {}",
                mpath.display()
            );
            return ExitCode::from(1);
        }
        if let Some(csha) = &clouds_sha256 {
            if let Some((expected_clouds_sha, _)) = entries
                .iter()
                .find(|(_, name)| name == "carbon-clouds.json")
            {
                if expected_clouds_sha != csha {
                    eprintln!(
                        "error: carbon-clouds.json SHA-256 ({csha}) does not match signed manifest ({expected_clouds_sha})"
                    );
                    return ExitCode::from(1);
                }
            }
        }
        manifest_sha256 = Some(sha256_lower_hex(&manifest_bytes));
        manifest_verified = true;
    } else if must_verify_manifest {
        eprintln!(
            "error: signed release manifest SHA256SUMS.txt is missing (pass --manifest <PATH> or --binary-only for uninstalled dev builds)"
        );
        return ExitCode::from(1);
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
            "binary_only": binary_only,
            "version": CARBON_PKG_VERSION,
            "release_tag": CARBON_RELEASE_TAG,
            "git_sha": CARBON_GIT_SHA,
            "build_target": CARBON_BUILD_TARGET,
            "build_profile": CARBON_BUILD_PROFILE,
            "executable_path": exe_path,
            "executable_bytes": exe_bytes.len(),
            "executable_sha256": exe_sha256,
            "signing_pub_fingerprint_sha256": embedded_pub_sha,
            "test_pubkey_override": using_test_pubkey,
            "manifest_ecdsa_verified": manifest_verified,
            "manifest_sha256": manifest_sha256,
            "matched_binary_asset": matched_binary_asset,
            "clouds_manifest_verified": clouds_verified,
            "clouds_manifest_sha256": clouds_sha256,
            "kek_pins_verified": kek_pins_verified,
        }))
        .unwrap_or_default()
    );
    ExitCode::SUCCESS
}

fn execute_positional_csv_shorthand(csv_path: PathBuf, global_quiet: bool) -> ExitCode {
    let is_stdin_dash = csv_path.as_os_str() == "-";
    let lower_str = csv_path.to_string_lossy().to_ascii_lowercase();
    let looks_like_tabular_path = is_stdin_dash
        || lower_str.ends_with(".csv")
        || lower_str.ends_with(".tsv")
        || lower_str.ends_with(".jsonl")
        || lower_str.ends_with(".ndjson")
        || lower_str.ends_with(".gz")
        || lower_str.ends_with(".zst")
        || csv_path.to_string_lossy().contains('/')
        || csv_path.to_string_lossy().contains('\\');

    if !is_stdin_dash && !csv_path.is_file() {
        if looks_like_tabular_path {
            eprintln!("error: CSV file '{}' does not exist", csv_path.display());
        } else {
            eprintln!(
                "error: unrecognized subcommand '{}' (run `carbon --help` for commands, or pass an existing `.csv` file path)",
                csv_path.display()
            );
        }
        return ExitCode::from(2);
    }

    let clouds_override = env::var("CARBON_CLOUDS_MANIFEST")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    let resolved = match resolve_cli_target_and_auth(
        Vec::new(),
        None,
        clouds_override.as_deref(),
        None,
        "us-east-2",
        None,
        None,
        "cli_tenant",
        None,
        None,
        CarbonTierWire::Pro,
    ) {
        Ok(r) => r,
        Err((code, msg)) => {
            eprintln!("error: {msg}");
            return code;
        }
    };
    let show_stderr_progress = !global_quiet && std::io::stderr().is_terminal();
    let opts = CarbonCliRunOptions {
        endpoints: resolved.endpoints,
        csv_path: Some(csv_path),
        target_column: "y".to_owned(),
        synthetic_rows: 240,
        synthetic_features: 6,
        tenant_id: resolved.tenant_id,
        tier: resolved.tier,
        effort: Effort::Fast,
        objective: "rmse".to_owned(),
        seed: 42,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 5,
        auth_token: resolved.auth_token,
        shred_after_run: true,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks: resolved.pinned_keks,
        pre_probed: Some((resolved.probed_client, resolved.first_probe)),
        target_was_explicit: false,
        show_stderr_progress,
        ..CarbonCliRunUxConfig::default()
    };
    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => match serde_json::to_string_pretty(&receipt) {
            Ok(s) => {
                println!("{s}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: serialize receipt: {e}");
                ExitCode::from(1)
            }
        },
        Err(err) => {
            eprintln!("error: carbon CSV shorthand failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn execute_doctor_command(
    cloud_arg: &str,
    clouds_manifest: Option<&str>,
    offline: bool,
) -> ExitCode {
    let mut healthy = true;
    let mut checks = Vec::new();
    let tty_stderr = std::io::stderr().is_terminal();

    let roles = compiled_binary_roles();
    checks.push(serde_json::json!({
        "check": "binary",
        "passed": true,
        "version": CARBON_PKG_VERSION,
        "release_tag": CARBON_RELEASE_TAG,
        "git_sha": CARBON_GIT_SHA,
        "build_target": CARBON_BUILD_TARGET,
        "build_profile": CARBON_BUILD_PROFILE,
        "roles": roles,
    }));
    if tty_stderr {
        let sha12: String = CARBON_GIT_SHA.chars().take(12).collect();
        eprintln!(
            "[doctor] PASS binary: carbon v{} ({}, git {}, target {}, roles: {})",
            CARBON_PKG_VERSION,
            CARBON_RELEASE_TAG,
            sha12,
            CARBON_BUILD_TARGET,
            roles.join(", ")
        );
    }

    let (config_dir_str, config_dir_ok, cred_mode_ok, cached_manifest_exists) =
        match verified_absolute_jopedime_config_dir() {
            Ok(dir) => {
                #[cfg(unix)]
                let (dir_mode_ok, dir_mode_octal) = {
                    use std::os::unix::fs::PermissionsExt;
                    let mut ok = true;
                    let mut octal = None;
                    if dir.exists() {
                        if let Ok(meta) = fs::metadata(&dir) {
                            let m = meta.permissions().mode() & 0o777;
                            octal = Some(format!("{m:04o}"));
                            if (m & 0o077) != 0 {
                                ok = false;
                            }
                        }
                    }
                    (ok, octal)
                };
                #[cfg(not(unix))]
                let (dir_mode_ok, dir_mode_octal): (bool, Option<String>) = (true, None);

                let cred_path = dir.join("carbon-credentials.json");
                #[cfg(unix)]
                let (cred_ok, cred_mode_octal) = {
                    use std::os::unix::fs::PermissionsExt;
                    let mut ok = true;
                    let mut octal = None;
                    if cred_path.is_file() {
                        if let Ok(meta) = fs::metadata(&cred_path) {
                            let m = meta.permissions().mode() & 0o777;
                            octal = Some(format!("{m:04o}"));
                            ok = m == 0o600;
                        }
                    }
                    (ok, octal)
                };
                #[cfg(not(unix))]
                let (cred_ok, cred_mode_octal): (bool, Option<String>) = {
                    let _ = &cred_path;
                    (true, None)
                };
                if !dir_mode_ok || !cred_ok {
                    healthy = false;
                }
                let cached_clouds = dir.join("carbon-clouds.json").is_file();
                checks.push(serde_json::json!({
                    "check": "config_security",
                    "passed": dir_mode_ok && cred_ok,
                    "config_dir": dir.display().to_string(),
                    "config_dir_exists": dir.exists(),
                    "config_dir_mode": dir_mode_octal,
                    "credentials_present": cred_path.is_file(),
                    "credentials_mode": cred_mode_octal,
                    "local_clouds_manifest_present": cached_clouds,
                }));
                if tty_stderr {
                    let tag = if dir_mode_ok && cred_ok {
                        "PASS"
                    } else {
                        "FAIL"
                    };
                    eprintln!(
                        "[doctor] {tag} config_security: dir={} (mode={:?}, credentials_mode={:?}, local_pins={cached_clouds})",
                        dir.display(),
                        dir_mode_octal.as_deref().unwrap_or("absent"),
                        cred_mode_octal.as_deref().unwrap_or("absent"),
                    );
                }
                (
                    dir.display().to_string(),
                    dir_mode_ok,
                    cred_ok,
                    cached_clouds,
                )
            }
            Err(err) => {
                healthy = false;
                checks.push(serde_json::json!({
                    "check": "config_security",
                    "passed": false,
                    "error": err,
                }));
                if tty_stderr {
                    eprintln!("[doctor] FAIL config_security: {err}");
                }
                (String::new(), false, false, false)
            }
        };

    let now_secs = now_epoch_ms() / 1_000;
    let mut stored_summary = BTreeMap::new();
    match list_stored_cloud_tokens() {
        Ok(map) => {
            for (cname, cred) in map {
                let expired = now_secs > cred.exp_epoch_secs;
                let ttl_rem = cred.exp_epoch_secs.saturating_sub(now_secs);
                stored_summary.insert(
                    cname.clone(),
                    serde_json::json!({
                        "tenant_id": cred.tenant_id,
                        "tier": cred.tier,
                        "expired": expired,
                        "ttl_remaining_secs": ttl_rem,
                    }),
                );
                if tty_stderr {
                    eprintln!(
                        "[doctor] INFO stored_token[{cname}]: tenant={} tier={:?} expired={expired} ttl_remaining={}s",
                        cred.tenant_id, cred.tier, ttl_rem
                    );
                }
            }
        }
        Err(err) => {
            healthy = false;
            checks.push(serde_json::json!({
                "check": "credentials_parse",
                "passed": false,
                "error": err,
            }));
            if tty_stderr {
                eprintln!("[doctor] FAIL credentials_parse: {err}");
            }
        }
    }

    let clouds_to_check: Vec<&str> = match cloud_arg.trim().to_ascii_lowercase().as_str() {
        "all" => vec!["prod", "dev"],
        "prod" => vec!["prod"],
        "dev" => vec!["dev"],
        other => {
            eprintln!("error: unsupported --cloud '{other}' for `carbon doctor` (expected prod, dev, or all)");
            return ExitCode::from(2);
        }
    };

    let mut cloud_reports = BTreeMap::new();
    for cname in clouds_to_check {
        match resolve_public_cloud_endpoint(cname, clouds_manifest) {
            Ok(entry) => {
                let candidates = entry.candidate_endpoints();
                let pins = entry.pinned_keks_by_endpoint();
                let all_pinned = !candidates.is_empty() && pins.len() == candidates.len();
                if !all_pinned {
                    healthy = false;
                }
                if offline {
                    cloud_reports.insert(
                        cname.to_owned(),
                        serde_json::json!({
                            "passed": all_pinned,
                            "configured_regions": candidates.len(),
                            "kek_pins_count": pins.len(),
                            "offline": true,
                        }),
                    );
                    if tty_stderr {
                        let tag = if all_pinned { "PASS" } else { "FAIL" };
                        eprintln!(
                            "[doctor] {tag} cloud[{cname}]: {} regional endpoints, {} P-256 KEK pins verified (offline)",
                            candidates.len(),
                            pins.len()
                        );
                    }
                } else {
                    match CarbonThinClient::connect_fastest_pinned(candidates.clone(), &pins) {
                        Ok((probed_client, probed)) => {
                            let remote_ver = probed_client.fetch_server_version().ok().flatten();
                            let fastest = probed.first();
                            let probed_list: Vec<serde_json::Value> = probed
                                .iter()
                                .map(|p| {
                                    serde_json::json!({
                                        "region": p.region,
                                        "endpoint": p.base_url,
                                        "rtt_ms": p.rtt_ms,
                                    })
                                })
                                .collect();
                            let cloud_ok = all_pinned && !probed.is_empty();
                            if !cloud_ok {
                                healthy = false;
                            }
                            cloud_reports.insert(
                                cname.to_owned(),
                                serde_json::json!({
                                    "passed": cloud_ok,
                                    "configured_regions": candidates.len(),
                                    "healthy_regions": probed.len(),
                                    "kek_pins_verified": all_pinned,
                                    "fastest_region": fastest.map(|f| f.region.clone()),
                                    "fastest_rtt_ms": fastest.map(|f| f.rtt_ms),
                                    "server_git_sha": remote_ver.as_ref().map(|v| v.git_sha.clone()),
                                    "server_release_id": remote_ver.as_ref().map(|v| v.release_id.clone()),
                                    "regions": probed_list,
                                }),
                            );
                            if tty_stderr {
                                if let Some(f) = fastest {
                                    eprintln!(
                                        "[doctor] PASS cloud[{cname}]: {}/{} healthy regions | fastest={} ({:.1} ms RTT) | KEK pins verified={all_pinned}",
                                        probed.len(),
                                        candidates.len(),
                                        f.region,
                                        f.rtt_ms
                                    );
                                }
                            }
                        }
                        Err(err) => {
                            healthy = false;
                            cloud_reports.insert(
                                cname.to_owned(),
                                serde_json::json!({
                                    "passed": false,
                                    "configured_regions": candidates.len(),
                                    "error": err,
                                }),
                            );
                            if tty_stderr {
                                eprintln!("[doctor] FAIL cloud[{cname}]: {err}");
                            }
                        }
                    }
                }
            }
            Err(err) => {
                healthy = false;
                cloud_reports.insert(
                    cname.to_owned(),
                    serde_json::json!({
                        "passed": false,
                        "error": err,
                    }),
                );
                if tty_stderr {
                    eprintln!("[doctor] FAIL cloud[{cname}] manifest: {err}");
                }
            }
        }
    }

    if tty_stderr {
        if stored_summary.is_empty() {
            eprintln!("[doctor] HINT: Run `carbon register --free` to mint a 30-day Free capability token (or run immediately in anonymous Free mode).");
        }
        eprintln!(
            "[doctor] OVERALL: {}",
            if healthy {
                "HEALTHY"
            } else {
                "DEGRADED / ACTION REQUIRED"
            }
        );
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "healthy": healthy,
            "version": CARBON_PKG_VERSION,
            "release_tag": CARBON_RELEASE_TAG,
            "git_sha": CARBON_GIT_SHA,
            "build_target": CARBON_BUILD_TARGET,
            "build_profile": CARBON_BUILD_PROFILE,
            "config_dir": config_dir_str,
            "config_dir_permissions_ok": config_dir_ok,
            "credentials_permissions_ok": cred_mode_ok,
            "local_clouds_manifest_present": cached_manifest_exists,
            "default_clouds_url": DEFAULT_PUBLIC_CLOUDS_URL,
            "fallback_clouds_url": FALLBACK_PUBLIC_CLOUDS_URL,
            "stored_credentials": stored_summary,
            "clouds": cloud_reports,
            "checks": checks,
        }))
        .unwrap_or_default()
    );

    if healthy {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

#[cfg(feature = "engine")]
fn run_engine_runtime(runtime_api: &str, work_dir: PathBuf, is_aws_prod: bool) -> ExitCode {
    let Ok(cas_url) = env::var("CARBON_CAS_URL") else {
        eprintln!("error: CARBON_CAS_URL is required for `carbon engine`");
        return ExitCode::from(2);
    };
    let region = match env::var("CARBON_REGION").or_else(|_| env::var("AWS_REGION")) {
        Ok(val) if !val.trim().is_empty() => val,
        _ => "us-east-2".to_owned(),
    };
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let eval_invoker = env::var("CARBON_EVAL_INVOKE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .map(|u| open_eval_invoker(&u, &region));
    let (seal_key, engine_kek) = match resolve_engine_secrets(&region, is_aws_prod) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    };

    let cas = match open_remote_cas(&cas_url, &region, &internal_token) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("error: open_remote_cas: {err}");
            return ExitCode::from(1);
        }
    };
    let force_hybrid = env::var("CARBON_FORCE_HYBRID_FANOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let worker = match CarbonEngineWorker::new(cas, eval_invoker, seal_key, work_dir) {
        Ok(w) => w
            .with_engine_kek(engine_kek.map(Arc::new))
            .with_forced_hybrid_fanout(force_hybrid),
        Err(err) => {
            eprintln!("error: init CarbonEngineWorker: {err}");
            return ExitCode::from(1);
        }
    };
    match run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_invocation_bytes(&inv.payload, inv.deadline_epoch_ms)
    }) {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: carbon engine runtime loop failed: {err}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "eval")]
fn run_eval_runtime(runtime_api: &str, scratch_dir: PathBuf) -> ExitCode {
    let Ok(cas_url) = env::var("CARBON_CAS_URL") else {
        eprintln!("error: CARBON_CAS_URL is required for `carbon eval`");
        return ExitCode::from(2);
    };
    let region = match env::var("CARBON_REGION").or_else(|_| env::var("AWS_REGION")) {
        Ok(val) if !val.trim().is_empty() => val,
        _ => "us-east-2".to_owned(),
    };
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let cas = match open_remote_cas(&cas_url, &region, &internal_token) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("error: open_remote_cas: {err}");
            return ExitCode::from(1);
        }
    };
    let worker = match CarbonEvalWorker::new(cas, scratch_dir) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("error: init CarbonEvalWorker: {err}");
            return ExitCode::from(1);
        }
    };
    match run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_wire_bytes(&inv.payload)
    }) {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: carbon eval runtime loop failed: {err}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "emu")]
fn run_emu_self_test(work_dir: PathBuf) -> Result<(), String> {
    fs::create_dir_all(&work_dir).map_err(|e| format!("create work_dir: {e}"))?;
    let mut emu = CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(work_dir))?;
    let pro_tok = emu.mint_capability_token("emu_tenant", CarbonTierWire::Pro, false)?;
    let (raw_client, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()])?;
    let client = raw_client.with_capability_token(pro_tok);
    let mut rows = Vec::with_capacity(60);
    for i in 0..60 {
        let x1 = (i as f64) * 0.25 - 7.0;
        let x2 = ((i % 7) as f64) * 0.5;
        let y = 2.5 * x1 - 1.2 * x2 + 0.75;
        rows.push(vec![x1, x2, y]);
    }
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &rows,
        "y",
        42,
        0.20,
    )?;
    client.upload_prepared("emu_tenant", CarbonTierWire::Pro, &prepared)?;
    let run_id = "run_01JCCARB0NEM00000000000001"
        .parse()
        .map_err(|_| "invalid self-test RunId".to_owned())?;
    let ds_id = "ds_111111111111111111111111"
        .parse()
        .map_err(|_| "invalid self-test DatasetId".to_owned())?;
    let progress = client.start_run(
        run_id,
        ds_id,
        "emu_tenant",
        CarbonTierWire::Pro,
        Effort::Fast,
        "rmse",
        42,
        &prepared,
    )?;
    let pred = client.predict(
        "emu_tenant",
        "run_01JCCARB0NEM00000000000001",
        vec!["x1".to_owned(), "x2".to_owned()],
        vec![vec![Some(1.0), Some(2.0)]],
    )?;
    let shred = client.shred_run("emu_tenant", "run_01JCCARB0NEM00000000000001")?;
    let telemetry = emu.telemetry();
    emu.shutdown();
    println!(
        "{}",
        serde_json::json!({
            "self_test": "ok",
            "probed_region": probed[0].region,
            "encrypted_cuen": progress.encrypted_cuen,
            "state": progress.state,
            "predictions_count": pred.row_count,
            "shredded": shred.shredded,
            "edge_invocations": telemetry.edge_invocations,
            "engine_invocations": telemetry.engine_invocations,
            "eval_invocations": telemetry.eval_invocations,
        })
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_quickstart_command(
    emit_csv: Option<PathBuf>,
    csv_only: bool,
    cloud: String,
    clouds_manifest: Option<String>,
    endpoint: Vec<String>,
    effort_raw: &str,
    no_shred: bool,
    quiet: bool,
) -> ExitCode {
    let effort_val = match parse_api_effort(effort_raw) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::from(2);
        }
    };

    if let Some(csv_out) = &emit_csv {
        let (_, rows, _) = generate_nonlinear_benchmark_rows(240, 6, 42);
        let mut buf = String::with_capacity(16 * 1024);
        buf.push_str("feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,target\n");
        for r in &rows {
            for (idx, val) in r.iter().enumerate() {
                if idx > 0 {
                    buf.push(',');
                }
                buf.push_str(&format!("{val:.6}"));
            }
            buf.push('\n');
        }
        if let Some(parent) = csv_out.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(err) = fs::write(csv_out, buf.as_bytes()) {
            eprintln!("error: write sample CSV {}: {err}", csv_out.display());
            return ExitCode::from(1);
        }
        if !quiet {
            eprintln!(
                "[quickstart] Wrote 240-row x 6-feature sample CSV -> {} (target column: 'target')",
                csv_out.display()
            );
        }
        if csv_only {
            println!(
                "{}",
                serde_json::json!({
                    "written_csv": csv_out,
                    "rows": 240,
                    "features": 6,
                    "target_column": "target",
                })
            );
            return ExitCode::SUCCESS;
        }
    }

    let cloud_opt = if endpoint.is_empty() {
        Some(cloud)
    } else {
        None
    };
    let resolved = match resolve_cli_target_and_auth(
        endpoint,
        cloud_opt,
        clouds_manifest.as_deref(),
        None,
        "us-east-2",
        None,
        None,
        "quickstart_user",
        None,
        None,
        CarbonTierWire::Free,
    ) {
        Ok(r) => r,
        Err((code, msg)) => {
            eprintln!("error: {msg}");
            return code;
        }
    };

    if !quiet {
        eprintln!(
            "[quickstart] Carbon {} — Zero-Knowledge E2E Walkthrough",
            CARBON_PKG_VERSION,
        );
        eprintln!(
            "[quickstart] Step 1/4: Connected to fastest regional edge '{}' ({:.1} ms RTT, P-256 KEK verified)",
            resolved.first_probe.region, resolved.first_probe.rtt_ms
        );
        eprintln!(
            "[quickstart] Step 2/4: Conditioning & encrypting 240x6 tabular dataset client-side (CUEN v1 + CUDK v1)..."
        );
    }

    let opts = CarbonCliRunOptions {
        endpoints: resolved.endpoints,
        csv_path: emit_csv.clone(),
        target_column: if emit_csv.is_some() {
            "target".to_owned()
        } else {
            "y".to_owned()
        },
        synthetic_rows: 240,
        synthetic_features: 6,
        tenant_id: resolved.tenant_id,
        tier: resolved.tier,
        effort: effort_val,
        objective: "rmse".to_owned(),
        seed: 42,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 5,
        auth_token: resolved.auth_token,
        shred_after_run: !no_shred,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks: resolved.pinned_keks,
        pre_probed: Some((resolved.probed_client, resolved.first_probe)),
        target_was_explicit: emit_csv.is_some(),
        show_stderr_progress: !quiet,
        ..CarbonCliRunUxConfig::default()
    };

    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => {
            if !quiet {
                eprintln!(
                    "[quickstart] Step 3/4: Holdout R^2 = {:.4} | 5 encrypted /predict rows decrypted locally",
                    receipt.holdout_r2.unwrap_or(0.0)
                );
                eprintln!(
                    "[quickstart] Step 4/4: Remote S3/CAS blobs cryptographically shredded = {} | Next: `carbon register --free`",
                    receipt.shredded_after_run
                );
            }
            match serde_json::to_string_pretty(&receipt) {
                Ok(s) => {
                    println!("{s}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: serialize receipt: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Err(err) => {
            eprintln!("error: carbon quickstart failed: {err}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "engine")]
fn resolve_lambda_runtime_api() -> (Result<String, env::VarError>, bool) {
    let aws = env::var("AWS_LAMBDA_RUNTIME_API");
    let is_aws_prod = aws.is_ok();
    (aws.or_else(|_| env::var("LAMBDA_RUNTIME_API")), is_aws_prod)
}

fn write_optional_outputs(
    json_out: Option<PathBuf>,
    markdown_out: Option<PathBuf>,
    json_text: &str,
    md_text: &str,
) -> Result<(), String> {
    if let Some(p) = json_out {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&p, format!("{json_text}\n"))
            .map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    if let Some(p) = markdown_out {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&p, md_text).map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    const CLI_MAIN_STACK_BYTES: usize = 16 * 1024 * 1024;
    match std::thread::Builder::new()
        .name("carbon-cli".to_owned())
        .stack_size(CLI_MAIN_STACK_BYTES)
        .spawn(run_cli_main)
    {
        Ok(handle) => match handle.join() {
            Ok(code) => code,
            Err(_) => ExitCode::from(101),
        },
        Err(_) => run_cli_main(),
    }
}

#[inline(never)]
fn run_cli_main() -> ExitCode {
    let cli = Cli::parse();
    let log_format = match cli.log_format.trim().to_ascii_lowercase().as_str() {
        "text" | "" => CarbonLogFormat::Text,
        "json" => CarbonLogFormat::Json,
        other => {
            eprintln!("error: unsupported --log-format `{other}` (expected `text` or `json`)");
            return ExitCode::from(2);
        }
    };
    let color_mode = match cli.color.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => CarbonColorMode::Auto,
        "always" => CarbonColorMode::Always,
        "never" => CarbonColorMode::Never,
        other => {
            eprintln!(
                "error: unsupported --color `{other}` (expected `auto`, `always`, or `never`)"
            );
            return ExitCode::from(2);
        }
    };
    init_cli_observability(cli.verbose, cli.quiet, log_format, color_mode);
    let _sigint_flag = install_sigint_flag();
    let global_quiet = cli.quiet;

    #[cfg(feature = "engine")]
    if cli.csv_file.is_none() && cli.command.is_none() {
        let (rt_res, is_aws_prod) = resolve_lambda_runtime_api();
        if let Ok(runtime_api) = rt_res {
            let work_dir = match env::var("CARBON_WORK_DIR") {
                Ok(val) if !val.trim().is_empty() => PathBuf::from(val),
                _ => env::temp_dir().join("carbon-lambda"),
            };
            let role = env::var("CARBON_ROLE").unwrap_or_default();
            return match role.trim() {
                "engine" => run_engine_runtime(&runtime_api, work_dir, is_aws_prod),
                #[cfg(feature = "eval")]
                "eval" => run_eval_runtime(&runtime_api, work_dir),
                other => {
                    eprintln!(
                        "error: CARBON_ROLE='{other}' unsupported for `carbon` bootstrap (expected 'engine' or 'eval')"
                    );
                    ExitCode::from(2)
                }
            };
        }
    }

    if let Err(tls_err) = jope_carbon::tls::validate_tls_env_config() {
        return emit_cli_error(false, &tls_err);
    }

    if let Some(csv_path) = cli.csv_file {
        return execute_positional_csv_shorthand(csv_path, global_quiet);
    }

    match cli.command.unwrap_or(Command::Version) {
        Command::Version => {
            println!(
                "{}",
                serde_json::json!({
                    "binary": "carbon",
                    "version": CARBON_PKG_VERSION,
                    "release_tag": CARBON_RELEASE_TAG,
                    "git_sha": CARBON_GIT_SHA,
                    "build_target": CARBON_BUILD_TARGET,
                    "build_profile": CARBON_BUILD_PROFILE,
                    "thin_cli_only": !cfg!(feature = "engine"),
                    "roles": compiled_binary_roles(),
                })
            );
            ExitCode::SUCCESS
        }
        Command::SelfVerify {
            manifest,
            signature,
            clouds_manifest,
            allow_test_pubkey,
            require_manifest,
            binary_only,
        } => execute_self_verify_command(
            manifest,
            signature,
            clouds_manifest,
            allow_test_pubkey,
            require_manifest,
            binary_only,
        ),
        #[cfg(feature = "engine")]
        Command::Package {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            jobs,
            include_cli_binaries,
        } => match execute_carbon_package(&CarbonPackageOptions {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            build_jobs: jobs,
            include_cli_binaries,
        }) {
            Ok(manifest) => match serde_json::to_string_pretty(&manifest) {
                Ok(s) => {
                    println!("{s}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: serialize manifest: {e}");
                    ExitCode::from(1)
                }
            },
            Err(err) => {
                eprintln!("error: carbon package failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::Deploy {
            manifest,
            environment,
            regions,
            artifacts_bucket,
            bootstrap_secrets,
            decommission_unstacked_prototype,
            dry_run,
            skip_redteam,
            json_out,
        } => match execute_carbon_deploy(&CarbonDeployOptions {
            manifest_path: manifest,
            environment,
            regions,
            artifacts_bucket_override: artifacts_bucket,
            bootstrap_secrets,
            decommission_unstacked_prototype,
            dry_run,
            skip_redteam,
        }) {
            Ok(report) => {
                let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                    return ExitCode::from(1);
                };
                if let Err(e) = write_optional_outputs(json_out, None, &json_text, "") {
                    eprintln!("error: {e}");
                    return ExitCode::from(1);
                }
                println!("{json_text}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon deploy failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::Rollback {
            environment,
            regions,
            to_versions,
        } => {
            let target_versions = if to_versions.is_empty() {
                None
            } else if to_versions.len() == 3 {
                Some(CarbonLiveAliasVersions {
                    edge_version: to_versions[0].clone(),
                    engine_version: to_versions[1].clone(),
                    eval_version: to_versions[2].clone(),
                })
            } else {
                eprintln!("error: --to-versions expects exactly 3 comma-separated versions (<edge>,<engine>,<eval>)");
                return ExitCode::from(2);
            };
            match execute_carbon_rollback_multi(&environment, &regions, target_versions) {
                Ok(mut receipts) => {
                    let s = if receipts.len() == 1 {
                        serde_json::to_string_pretty(&receipts.remove(0))
                    } else {
                        serde_json::to_string_pretty(&receipts)
                    };
                    if let Ok(text) = s {
                        println!("{text}");
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon rollback failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        #[cfg(feature = "engine")]
        Command::CloudsManifest {
            out,
            dev_regions,
            prod_regions,
        } => match generate_public_clouds_manifest(&dev_regions, &prod_regions) {
            Ok(manifest) => {
                let Ok(json_bytes) = serde_json::to_vec_pretty(&manifest) else {
                    eprintln!("error: serialize clouds manifest failed");
                    return ExitCode::from(1);
                };
                if let Some(parent) = out.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                if let Err(e) = fs::write(&out, &json_bytes) {
                    eprintln!("error: write {}: {e}", out.display());
                    return ExitCode::from(1);
                }
                println!("{}", String::from_utf8_lossy(&json_bytes));
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon clouds-manifest failed: {err}");
                ExitCode::from(1)
            }
        },
        Command::Register {
            cloud,
            free: _free,
            label,
            clouds_manifest,
            endpoint,
        } => {
            let (candidates, pinned_keks) = if let Some(ep) = endpoint {
                (vec![ep], BTreeMap::new())
            } else {
                match resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref()) {
                    Ok(entry) => (entry.candidate_endpoints(), entry.pinned_keks_by_endpoint()),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            };
            let mut client =
                match CarbonThinClient::connect_fastest_pinned(candidates, &pinned_keks) {
                    Ok((c, _)) => c,
                    Err(e) => {
                        eprintln!("error: probe '{cloud}' failed: {e}");
                        return ExitCode::from(1);
                    }
                };
            let reg = match client.register_free(label.as_deref()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: carbon register failed: {e}");
                    return ExitCode::from(1);
                }
            };
            let active_ep = client.active_endpoint().to_owned();
            let now_secs = now_epoch_ms() / 1_000;
            match save_stored_cloud_token(&cloud, &active_ep, &reg.capability_token) {
                Ok(saved) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "registered": true,
                            "cloud": cloud.trim().to_ascii_lowercase(),
                            "endpoint": active_ep,
                            "tenant_id": saved.tenant_id,
                            "tier": saved.tier,
                            "expires_epoch_secs": reg.expires_epoch_secs.0,
                            "ttl_remaining_secs": reg.expires_epoch_secs.0.saturating_sub(now_secs),
                            "credentials_saved": true,
                        }))
                        .unwrap_or_default()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Login {
            cloud,
            clouds_manifest,
            endpoint,
            token,
            token_file,
        } => {
            let raw_tok = match read_cli_token_source(token, token_file) {
                Ok(Some(t)) => t,
                Ok(None) => match env::var("CARBON_AUTH_TOKEN") {
                    Ok(v) if !v.trim().is_empty() => v.trim().to_owned(),
                    _ => {
                        eprintln!(
                            "error: pass --token <ctk1...> or --token-file <PATH> to `carbon login`"
                        );
                        return ExitCode::from(2);
                    }
                },
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let claims = match CarbonCapabilityClaims::decode_unverified(&raw_tok) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let now_secs = now_epoch_ms() / 1_000;
            if now_secs > claims.exp_epoch_secs {
                eprintln!(
                    "error: capability token for tenant '{}' has already expired (exp={})",
                    claims.tenant_id, claims.exp_epoch_secs
                );
                return ExitCode::from(2);
            }
            let (candidates, pinned_keks) = if let Some(ep) = endpoint {
                (vec![ep], BTreeMap::new())
            } else {
                match resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref()) {
                    Ok(entry) => (entry.candidate_endpoints(), entry.pinned_keks_by_endpoint()),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            };
            let active_ep = match CarbonThinClient::connect_fastest_pinned(candidates, &pinned_keks)
            {
                Ok((client, _)) => client.active_endpoint().to_owned(),
                Err(e) => {
                    eprintln!("error: probe '{cloud}' failed: {e}");
                    return ExitCode::from(1);
                }
            };
            match save_stored_cloud_token(&cloud, &active_ep, &raw_tok) {
                Ok(saved) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "logged_in": true,
                            "cloud": cloud.trim().to_ascii_lowercase(),
                            "endpoint": active_ep,
                            "tenant_id": saved.tenant_id,
                            "tier": saved.tier,
                            "exp_epoch_secs": saved.exp_epoch_secs,
                            "ttl_remaining_secs": saved.exp_epoch_secs.saturating_sub(now_secs),
                        }))
                        .unwrap_or_default()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Whoami {
            cloud,
            clouds_manifest,
            token,
            token_file,
            json: _,
        } => {
            let now_secs = now_epoch_ms() / 1_000;
            let ep = resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref())
                .ok()
                .map(|e| e.public_endpoint);
            let inline = match read_cli_token_source(token, token_file) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let env_tok = env::var("CARBON_AUTH_TOKEN")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
            let (tok_opt, source_label) = if let Some(t) = inline {
                (Some(t), "cli_flag")
            } else if let Some(t) = env_tok {
                (Some(t), "env_CARBON_AUTH_TOKEN")
            } else if let Ok(Some(stored)) = load_stored_cloud_token(&cloud) {
                (Some(stored.capability_token), "stored_credentials")
            } else {
                (None, "anonymous_default")
            };
            if let Some(tok) = tok_opt {
                match CarbonCapabilityClaims::decode_unverified(&tok) {
                    Ok(claims) => {
                        let expired = now_secs > claims.exp_epoch_secs;
                        let _ = write_stdout_line(
                            &serde_json::to_string_pretty(&serde_json::json!({
                                "cloud": cloud.trim().to_ascii_lowercase(),
                                "endpoint": ep,
                                "auth_mode": if expired { "expired_token" } else { "capability_token" },
                                "source": source_label,
                                "tenant_id": claims.tenant_id,
                                "tier": claims.tier,
                                "admin": claims.admin,
                                "expired": expired,
                                "exp_epoch_secs": claims.exp_epoch_secs,
                                "ttl_remaining_secs": claims.exp_epoch_secs.saturating_sub(now_secs),
                            }))
                            .unwrap_or_default(),
                        );
                        if expired {
                            ExitCode::from(EXIT_AUTH_EXPIRED)
                        } else {
                            ExitCode::SUCCESS
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::from(2)
                    }
                }
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "cloud": cloud.trim().to_ascii_lowercase(),
                        "endpoint": ep,
                        "auth_mode": "anonymous_free",
                        "source": source_label,
                        "tier": "free",
                    }))
                    .unwrap_or_default()
                );
                ExitCode::SUCCESS
            }
        }
        Command::Logout { cloud, all } => {
            let target = if all {
                None
            } else {
                Some(cloud.as_deref().unwrap_or("prod"))
            };
            match remove_stored_cloud_token(target) {
                Ok(removed) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "logged_out": true,
                            "scope": target.unwrap_or("all"),
                            "entries_removed": removed,
                        }))
                        .unwrap_or_default()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Doctor {
            cloud,
            clouds_manifest,
            offline,
        } => execute_doctor_command(&cloud, clouds_manifest.as_deref(), offline),
        Command::Ping {
            mut endpoint,
            mut cloud,
            clouds_manifest,
            environment,
            region,
        } => {
            if endpoint.is_empty() && cloud.is_none() && environment.is_none() {
                if let Ok(env_ep) = env::var("CARBON_ENDPOINT") {
                    let trimmed = env_ep.trim();
                    if !trimmed.is_empty() {
                        endpoint.push(trimmed.to_owned());
                    }
                }
                if endpoint.is_empty() {
                    cloud = Some("prod".to_owned());
                }
            }
            let mut pinned_keks: BTreeMap<String, String> = BTreeMap::new();
            if let Some(cloud_name) = &cloud {
                match resolve_public_cloud_endpoint(cloud_name, clouds_manifest.as_deref()) {
                    Ok(entry) => {
                        pinned_keks = entry.pinned_keks_by_endpoint();
                        if endpoint.is_empty() {
                            endpoint = entry.candidate_endpoints();
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            if let Some(env_name) = &environment {
                if endpoint.is_empty() {
                    match resolve_stack_endpoint_if_compiled(env_name, &region) {
                        Ok(url) => endpoint.push(url),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if endpoint.is_empty() {
                eprintln!(
                    "error: pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>"
                );
                return ExitCode::from(2);
            }
            match CarbonThinClient::connect_fastest_pinned(endpoint, &pinned_keks) {
                Ok((_, probed)) => {
                    let Some(first) = probed.first() else {
                        eprintln!("error: empty probe response");
                        return ExitCode::from(1);
                    };
                    let probed_regions: Vec<serde_json::Value> = probed
                        .iter()
                        .map(|p| {
                            serde_json::json!({
                                "region": p.region,
                                "endpoint": p.base_url,
                                "rtt_ms": p.rtt_ms,
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "healthy": true,
                            "cloud": cloud,
                            "endpoint": first.base_url,
                            "region": first.region,
                            "rtt_ms": first.rtt_ms,
                            "enterprise_self_hosted": first.enterprise_self_hosted,
                            "engine_kek_pubkey_b64": first.engine_kek_pubkey_b64,
                            "kek_pin_verified": !pinned_keks.is_empty(),
                            "healthy_regions": probed_regions,
                        }))
                        .unwrap_or_default()
                    );
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon ping failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Run {
            csv,
            target,
            synthetic_rows,
            features,
            stop_after_ms,
            resume,
            predict_rows,
            save,
            format,
            shred,
            progress,
            common,
        } => {
            if global_quiet && progress {
                return emit_cli_error(common.json, "--quiet and --progress cannot be combined");
            }
            if shred && save.is_some() {
                return emit_cli_error(
                    common.json,
                    "--shred and --save cannot be combined (a shredded run cannot be queried by `carbon predict`)",
                );
            }
            let pretty_stdout = !common.json && format.trim().eq_ignore_ascii_case("pretty");
            let persist_handle = !common.dry_run
                && !shred
                && (save.is_some() || verified_absolute_jopedime_config_dir().is_ok());
            execute_train_command(
                "run",
                common,
                csv,
                target,
                synthetic_rows,
                features,
                stop_after_ms,
                resume,
                predict_rows,
                save,
                shred,
                persist_handle,
                pretty_stdout,
                progress,
                global_quiet,
            )
        }
        Command::Fit {
            csv,
            csv_flag,
            target,
            save,
            format,
            common,
        } => {
            let pretty_stdout = should_emit_pretty_stdout(&format, common.json);
            let persist_handle = !common.dry_run;
            execute_train_command(
                "fit",
                common,
                csv.or(csv_flag),
                target,
                240,
                6,
                None,
                false,
                5,
                save,
                false,
                persist_handle,
                pretty_stdout,
                false,
                global_quiet,
            )
        }
        Command::Predict {
            csv,
            csv_flag,
            model,
            run,
            out,
            id_column,
            include_inputs,
            no_header,
            delimiter,
            token,
            token_file,
            format,
            json,
        } => {
            let Some(csv_path) = csv.or(csv_flag) else {
                return emit_cli_error(json, "missing input CSV path");
            };
            let delim_override = match parse_delimiter_arg(delimiter.as_deref()) {
                Ok(d) => d,
                Err(e) => return emit_cli_error(json, &e),
            };
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            match execute_predict_csv_with_handle(
                &handle,
                &csv_path,
                out.as_deref(),
                delim_override,
                no_header,
                id_column.as_deref(),
                include_inputs,
                override_tok,
            ) {
                Ok(receipt) => {
                    let streamed_stdout_csv = out.as_deref().is_some_and(|p| p.as_os_str() == "-");
                    if streamed_stdout_csv {
                        ExitCode::SUCCESS
                    } else if should_emit_pretty_stdout(&format, json) {
                        println!("┌─ Carbon Zero-Knowledge Batch Prediction ───────────────────────────────┐");
                        println!(
                            "  Run & Region : {} | {} (CUEN\\x01={})",
                            receipt.run_id, receipt.region, receipt.encrypted_cuen
                        );
                        println!(
                            "  Scored Rows  : {} rows across {} encrypted batch(es) in {:.1} ms",
                            receipt.rows_scored, receipt.batches, receipt.wall_ms
                        );
                        if let Some(out_p) = &receipt.output_csv {
                            println!(
                                "  Output CSV   : {out_p} (sha256={})",
                                &receipt.predictions_sha256[..16]
                            );
                        }
                        if let Some(ev) = &receipt.evaluation_if_labeled {
                            let r2s = ev
                                .r2
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "n/a".to_owned());
                            println!(
                                "  Labeled Eval : R^2 = {r2s} | RMSE = {:.4} | MAE = {:.4} (p90={:.4})",
                                ev.rmse, ev.mae, ev.abs_err_p90
                            );
                        }
                        let preview: Vec<String> = receipt
                            .sample_predictions
                            .iter()
                            .take(5)
                            .map(|v| format!("{v:.4}"))
                            .collect();
                        println!("  Sample Preds : [{}]", preview.join(", "));
                        println!("└────────────────────────────────────────────────────────────────────────┘");
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&receipt) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize predict receipt: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon predict failed: {err}")),
            }
        }
        Command::Inspect {
            csv,
            csv_flag,
            target,
            no_header,
            delimiter,
            tier,
            format,
            json,
        } => {
            let Some(csv_path) = csv.or(csv_flag) else {
                return emit_cli_error(json, "missing input CSV path");
            };
            let delim_override = match parse_delimiter_arg(delimiter.as_deref()) {
                Ok(d) => d,
                Err(e) => return emit_cli_error(json, &e),
            };
            let Ok(ptier) = parse_product_tier(&tier) else {
                return emit_cli_error(json, &format!("invalid --tier `{tier}`"));
            };
            match inspect_tabular_file_with_tier(
                &csv_path,
                target.as_deref(),
                delim_override,
                !no_header,
                ptier,
            ) {
                Ok(report) => {
                    if should_emit_pretty_stdout(&format, json) {
                        let _ = write_stdout_line(&render_pretty_inspect_report(&report));
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&report) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize inspect report: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon inspect failed: {err}")),
            }
        }
        Command::Runs { json } => match list_saved_model_handles() {
            Ok(entries) => {
                if !json && std::io::stdout().is_terminal() {
                    if entries.is_empty() {
                        println!("No saved Carbon runs found. Train one with: carbon fit <CSV> --target <COL>");
                    } else {
                        println!(
                            "{:<30}  {:<12}  {:<14}  {:>8}  {:>10}  HANDLE",
                            "RUN_ID", "REGION", "TARGET", "FIT_ROWS", "HOLDOUT_R2"
                        );
                        for e in &entries {
                            let r2s = e
                                .holdout_r2
                                .map(|v| format!("{v:.5}"))
                                .unwrap_or_else(|| "-".to_owned());
                            println!(
                                "{:<30}  {:<12}  {:<14}  {:>8}  {:>10}  {}",
                                e.run_id, e.region, e.target_column, e.fit_rows, r2s, e.handle_path
                            );
                        }
                    }
                } else if let Ok(s) = serde_json::to_string_pretty(&entries) {
                    println!("{s}");
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: list saved runs failed: {err}");
                ExitCode::from(1)
            }
        },
        Command::Status {
            model,
            run,
            token,
            token_file,
            json,
        } => {
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            match query_handle_status(&handle, override_tok) {
                Ok(status) => {
                    if let Ok(s) = serde_json::to_string_pretty(&status) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => emit_cli_error(json, &format!("carbon status failed: {err}")),
            }
        }
        Command::Shred {
            model,
            run,
            token,
            token_file,
            json,
        } => {
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (handle, resolved_path) =
                match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                    Ok(pair) => pair,
                    Err(e) => return emit_cli_error(json, &e),
                };
            let explicit_opt = model.as_deref().or(Some(resolved_path.as_path()));
            match shred_handle_and_local_keys(&handle, explicit_opt, override_tok) {
                Ok((remote, local_deleted)) => {
                    let _ = write_stdout_line(
                        &serde_json::to_string_pretty(&serde_json::json!({
                            "run_id": remote.run_id,
                            "remote_shredded": remote.shredded,
                            "remote_objects_deleted": remote.objects_deleted,
                            "local_key_handle_shredded": local_deleted,
                        }))
                        .unwrap_or_default(),
                    );
                    ExitCode::SUCCESS
                }
                Err(err) => emit_cli_error(json, &format!("carbon shred failed: {err}")),
            }
        }
        Command::Fitness { action } => execute_fitness_command(action),
        Command::Explain {
            model,
            run,
            top_k,
            token,
            token_file,
            format,
            json,
        } => {
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.explain_run(&handle.tenant_id, &handle.run_id, top_k) {
                Ok(resp) => {
                    if should_emit_pretty_stdout(&format, json) {
                        let _ = write_stdout_line(&render_pretty_explain_response(&resp));
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&resp) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize explain response: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon explain failed: {err}")),
            }
        }
        Command::Export {
            model,
            run,
            lang,
            out,
            token,
            token_file,
            json,
        } => {
            let lang_wire = match parse_export_lang(&lang) {
                Ok(l) => l,
                Err(e) => return emit_cli_error(json, &e),
            };
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.export_run(&handle.tenant_id, &handle.run_id, lang_wire) {
                Ok(resp) => {
                    let src_code = resp.source_code.clone().unwrap_or_default();
                    if let Some(out_path) = &out {
                        if out_path.as_os_str() == "-" {
                            let _ = write_stdout_line(src_code.trim_end_matches('\n'));
                            return ExitCode::SUCCESS;
                        }
                        if let Some(parent) = out_path.parent() {
                            if !parent.as_os_str().is_empty() {
                                let _ = fs::create_dir_all(parent);
                            }
                        }
                        if let Err(e) = fs::write(out_path, src_code.as_bytes()) {
                            return emit_cli_error(
                                json,
                                &format!("write export source {}: {e}", out_path.display()),
                            );
                        }
                        if json {
                            if let Ok(s) = serde_json::to_string_pretty(&resp) {
                                let _ = write_stdout_line(&s);
                            }
                        } else {
                            let _ = write_stdout_line(
                                &serde_json::to_string_pretty(&serde_json::json!({
                                    "run_id": resp.run_id,
                                    "lang": resp.lang,
                                    "output_path": out_path,
                                    "source_sha256": resp.source_sha256,
                                    "bytes": src_code.len(),
                                    "encrypted_cuen": resp.encrypted_export_cuen_b64.is_some(),
                                }))
                                .unwrap_or_default(),
                            );
                        }
                        ExitCode::SUCCESS
                    } else if json {
                        match serde_json::to_string_pretty(&resp) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(true, &format!("serialize export response: {e}"))
                            }
                        }
                    } else {
                        let _ = write_stdout_line(src_code.trim_end_matches('\n'));
                        ExitCode::SUCCESS
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon export failed: {err}")),
            }
        }
        Command::Resume {
            model,
            run,
            effort,
            additional_waves,
            token,
            token_file,
            json,
        } => {
            let effort_opt = match effort.as_deref() {
                Some(raw) => match parse_api_effort(raw) {
                    Ok(e) => Some(e),
                    Err(msg) => return emit_cli_error(json, &msg),
                },
                None => None,
            };
            let override_tok = match read_cli_token_source(token, token_file) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, resolved_path) =
                match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                    Ok(pair) => pair,
                    Err(e) => return emit_cli_error(json, &e),
                };
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.resume_run(
                &handle.tenant_id,
                &handle.run_id,
                effort_opt,
                additional_waves,
            ) {
                Ok(progress) => {
                    if let Some(loss) = progress.outer_holdout_loss {
                        handle.holdout_rmse = Some(loss);
                        let explicit_save = model.as_deref().or(Some(resolved_path.as_path()));
                        let has_embedded = handle.capability_token.is_some();
                        let _ =
                            persist_model_handle_with_options(&handle, explicit_save, has_embedded);
                    }
                    if let Ok(s) = serde_json::to_string_pretty(&progress) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => emit_cli_error(json, &format!("carbon resume failed: {err}")),
            }
        }
        Command::Upgrade {
            check,
            force,
            version,
            base_url,
            allow_test_pubkey,
        } => {
            let using_test_pubkey = allow_test_pubkey.is_some();
            let pub_pem = if let Some(p) = allow_test_pubkey {
                match fs::read_to_string(&p) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: read --allow-test-pubkey {}: {e}", p.display());
                        return ExitCode::from(2);
                    }
                }
            } else {
                let canonical_pub_pem = canonical_embedded_release_signing_pub_pem();
                let fp = sha256_lower_hex(canonical_pub_pem.as_bytes());
                if fp != EXPECTED_SIGNING_PUB_SHA256 {
                    eprintln!("error: embedded signing key fingerprint mismatch");
                    return ExitCode::from(1);
                }
                canonical_pub_pem
            };
            match execute_self_upgrade_with_options(
                base_url.as_deref(),
                version.as_deref(),
                &pub_pem,
                check,
                force || using_test_pubkey,
            ) {
                Ok(receipt) => {
                    if let Ok(s) = serde_json::to_string_pretty(&receipt) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon upgrade failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Completions { shell } => match emit_shell_completions(&shell) {
            Ok(script) => {
                print!("{script}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(2)
            }
        },
        Command::Bench {
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
            suite,
            datasets_dir,
            csv_dir,
            effort,
            seed,
            free,
            pro,
            tier,
            token,
            token_file,
            tenant,
            json_out,
            markdown_out,
        } => {
            let explicit_tier = match resolve_explicit_tier_flag(free, pro, tier.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let resolved = match resolve_cli_target_and_auth(
                endpoint,
                cloud,
                clouds_manifest.as_deref(),
                environment.as_deref(),
                &region,
                explicit_tier,
                tenant,
                "bench_tenant",
                token,
                token_file,
                CarbonTierWire::Pro,
            ) {
                Ok(r) => r,
                Err((code, msg)) => {
                    eprintln!("error: {msg}");
                    return code;
                }
            };
            let suite_norm = suite
                .as_deref()
                .map(|s| s.trim().to_ascii_lowercase())
                .unwrap_or_else(|| {
                    if datasets_dir.is_some() || csv_dir.is_some() {
                        "standard".to_owned()
                    } else {
                        "ops".to_owned()
                    }
                });
            if suite_norm != "ops" {
                let Ok(effort_val) = parse_api_effort(&effort) else {
                    eprintln!("error: invalid --effort `{effort}`");
                    return ExitCode::from(2);
                };
                let sci_opts = CarbonScientificBenchmarkOptions {
                    endpoints: resolved.endpoints,
                    pinned_keks: resolved.pinned_keks,
                    tier: resolved.tier,
                    effort: effort_val,
                    tenant_id: resolved.tenant_id,
                    auth_token: resolved.auth_token,
                    suite: suite_norm,
                    datasets_dir,
                    csv_dir,
                    seed,
                };
                return match execute_scientific_benchmark_suite(
                    &sci_opts,
                    std::io::stderr().is_terminal(),
                ) {
                    Ok(report) => {
                        let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                            return ExitCode::from(1);
                        };
                        let md_text = render_scientific_benchmark_markdown(&report);
                        if let Err(e) =
                            write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                        {
                            eprintln!("error: {e}");
                            return ExitCode::from(1);
                        }
                        println!("{json_text}");
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: carbon scientific bench failed: {err}");
                        ExitCode::from(1)
                    }
                };
            }
            match execute_carbon_cli_benchmark_with_token(
                resolved.endpoints,
                resolved.tier,
                &resolved.tenant_id,
                resolved.auth_token,
            ) {
                Ok(report) => {
                    let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                        return ExitCode::from(1);
                    };
                    let md_text = render_benchmark_markdown(&report);
                    if let Err(e) =
                        write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                    {
                        eprintln!("error: {e}");
                        return ExitCode::from(1);
                    }
                    println!("{json_text}");
                    if report.all_passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(err) => {
                    eprintln!("error: carbon bench failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Redteam {
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
            json_out,
            markdown_out,
        } => {
            let resolved_ep = match (endpoint, cloud, environment) {
                (Some(ep), _, _) => ep,
                (None, Some(cloud_name), _) => {
                    match resolve_public_cloud_endpoint(&cloud_name, clouds_manifest.as_deref()) {
                        Ok(entry) => entry.public_endpoint,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                (None, None, Some(env_name)) => {
                    match resolve_stack_endpoint_if_compiled(&env_name, &region) {
                        Ok(url) => url,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                (None, None, None) => {
                    eprintln!(
                        "error: pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>"
                    );
                    return ExitCode::from(2);
                }
            };
            match execute_carbon_redteam_suite(&resolved_ep) {
                Ok(report) => {
                    let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                        return ExitCode::from(1);
                    };
                    let md_text = render_redteam_markdown(&report);
                    if let Err(e) =
                        write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                    {
                        eprintln!("error: {e}");
                        return ExitCode::from(1);
                    }
                    println!("{json_text}");
                    if report.all_passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(err) => {
                    eprintln!("error: carbon redteam failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::TokenMint {
            tenant,
            tier,
            ttl_secs,
            admin,
            environment,
            region,
        } => {
            if ttl_secs == 0 || ttl_secs > MAX_MINT_TOKEN_TTL_SECS {
                eprintln!(
                    "error: --ttl-secs must be within 1..={MAX_MINT_TOKEN_TTL_SECS} seconds (got {ttl_secs})"
                );
                return ExitCode::from(2);
            }
            let Ok(tier_wire) = parse_tier_wire(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            if let Some(env_name) = environment {
                return match mint_vault_token_if_compiled(
                    &env_name, &region, &tenant, tier_wire, ttl_secs, admin,
                ) {
                    Ok(tok) => {
                        println!("{tok}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::from(2)
                    }
                };
            }
            let Ok(secret_hex) = env::var("CARBON_AUTH_SECRET_HEX") else {
                eprintln!("error: pass --env <dev|staging|prod> or set CARBON_AUTH_SECRET_HEX");
                return ExitCode::from(2);
            };
            let verifier = match CarbonAuthVerifier::from_hex(&secret_hex) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let exp = (now_epoch_ms() / 1_000).saturating_add(ttl_secs);
            match verifier.mint_token(&CarbonCapabilityClaims {
                tenant_id: tenant,
                tier: tier_wire,
                exp_epoch_secs: exp,
                admin,
            }) {
                Ok(tok) => {
                    println!("{tok}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Plan {
            tier,
            effort,
            total_rows,
            fit_rows,
            features,
            ms_per_candidate,
        } => {
            let Ok(product_tier) = parse_product_tier(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            let effort_tier: CarbonEffortTier = match effort.parse() {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("error: {err} (hint: {})", err.guidance().repair_hint);
                    return ExitCode::from(2);
                }
            };
            let est_ms = ms_per_candidate
                .unwrap_or_else(|| estimate_ms_per_candidate_1vcpu(fit_rows, features));
            match plan_carbon_execution(&CarbonPlanInput {
                product_tier,
                effort: effort_tier,
                total_rows,
                fit_rows,
                features,
                est_ms_per_candidate_1vcpu: est_ms,
            }) {
                Ok(decision) => {
                    if let Ok(s) = serde_json::to_string_pretty(&decision) {
                        println!("{s}");
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err} (hint: {})", err.guidance().repair_hint);
                    ExitCode::from(2)
                }
            }
        }
        Command::SpotCheck {
            region,
            monthly_spend_usd,
            p20_vcpus,
            mean_vcpus,
            p95_vcpus,
            cold_start_rate,
            consecutive_months,
        } => {
            let sample = MonthlyWindowMetrics::for_region(
                &region,
                monthly_spend_usd,
                p20_vcpus,
                mean_vcpus,
                p95_vcpus,
                cold_start_rate,
            );
            let history = vec![sample; consecutive_months.max(1)];
            let verdict = evaluate_spot_hybrid_gates(&history, &SpotAdvisorThresholds::default());
            if let Ok(s) = serde_json::to_string_pretty(&verdict) {
                println!("{s}");
            }
            ExitCode::SUCCESS
        }
        #[cfg(feature = "serve")]
        Command::Serve {
            listen,
            data_dir,
            cas_url,
            allow_local_fs_dev,
            dev_ephemeral,
            region,
            enable_user_envelope,
        } => {
            let effective_cas_url = cas_url
                .or_else(|| env::var("CARBON_CAS_URL").ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
            let is_loopback = listen.ip().is_loopback();
            if !is_loopback && effective_cas_url.is_none() && !allow_local_fs_dev {
                eprintln!(
                    "error: non-loopback `carbon serve` requires --cas-url s3://... (or CARBON_CAS_URL), or explicit --allow-local-fs-dev"
                );
                return ExitCode::from(2);
            }
            let allow_ephemeral_secrets = is_loopback || dev_ephemeral;
            let mut cfg = CarbonEnterpriseServeConfig::enterprise_default(listen, data_dir);
            cfg.cas_url = effective_cas_url;
            cfg.allow_local_fs_dev = is_loopback || allow_local_fs_dev;
            cfg.region = region;
            cfg.enable_user_envelope = enable_user_envelope;
            cfg.seal_secret = match decode_or_ephemeral_32byte_secret(
                "CARBON_SEAL_SECRET_HEX",
                allow_ephemeral_secrets,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            cfg.presign_secret = match decode_or_ephemeral_32byte_secret(
                "CARBON_PRESIGN_SECRET_HEX",
                allow_ephemeral_secrets,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if let Ok(auth_hex) = env::var("CARBON_AUTH_SECRET_HEX") {
                if !auth_hex.trim().is_empty() {
                    match decode_required_32byte_hex("CARBON_AUTH_SECRET_HEX") {
                        Ok(s) => cfg.auth_secret = Some(s),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if let Err(err) = CarbonEnterpriseServer::run_until_ctrl_c(cfg) {
                eprintln!("error: carbon serve failed: {err}");
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        #[cfg(feature = "emu")]
        Command::Emu {
            work_dir,
            self_test,
        } => {
            if self_test {
                match run_emu_self_test(work_dir) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("error: carbon emu --self-test failed: {err}");
                        ExitCode::from(1)
                    }
                }
            } else {
                eprintln!("error: pass --self-test for automated emulator execution");
                ExitCode::from(2)
            }
        }
        #[cfg(feature = "emu")]
        Command::CiLive4Tier {
            source_sha,
            out_dir,
            require_live_pro,
            repo_root,
        } => {
            match jope_carbon::cli::spec_live_ci_4tier::execute_ci_live_4tier_docker(
                &jope_carbon::cli::spec_live_ci_4tier::LiveCi4TierOptions {
                    source_sha,
                    out_dir,
                    require_live_pro,
                    repo_root,
                },
            ) {
                Ok(receipt) => {
                    if let Ok(s) = serde_json::to_string_pretty(&serde_json::json!({
                        "status": "passed",
                        "all_passed": receipt.all_passed,
                        "source_sha": receipt.source_sha,
                        "runs": receipt.runs.len(),
                        "total_estimated_usd": receipt.total_estimated_usd,
                        "total_wall_secs": receipt.total_wall_secs,
                    })) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon ci-live-4tier failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Quickstart {
            emit_csv,
            csv_only,
            cloud,
            clouds_manifest,
            endpoint,
            effort,
            no_shred,
        } => run_quickstart_command(
            emit_csv,
            csv_only,
            cloud,
            clouds_manifest,
            endpoint,
            &effort,
            no_shred,
            global_quiet,
        ),
        #[cfg(feature = "emu")]
        Command::SpecReport {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
            live_ci_receipt,
            source_sha,
            repo_root,
        } => match execute_carbon_spec_report(&CarbonSpecReportOptions {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
            live_ci_receipt_in: live_ci_receipt,
            source_sha,
            repo_root,
        }) {
            Ok(html_path) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "ok",
                        "html_path": html_path,
                    })
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon spec-report failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "emu")]
        Command::SpecVerify {
            dir,
            expected_source_sha,
        } => match execute_carbon_spec_verify(&CarbonSpecVerifyOptions {
            dir,
            expected_source_sha,
        }) {
            Ok(receipt) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "verified",
                        "pdf_file": receipt.pdf_file,
                        "pdf_bytes": receipt.pdf_bytes,
                        "pdf_sha256": receipt.pdf_sha256,
                        "pdf_page_count": receipt.pdf_page_count,
                        "screenshots_verified": receipt.screenshots.len(),
                        "source_sha": receipt.source_sha,
                    })
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon spec-verify failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::CoverageAudit {
            clouds_manifest,
            signing_pub,
            json_out,
        } => match jope_carbon::execute_live_aws_product_coverage_audit(
            &clouds_manifest,
            &signing_pub,
            json_out.as_deref(),
        ) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": if report.all_passed { "passed" } else { "failed" },
                        "all_passed": report.all_passed,
                        "total_checks_per_cloud": report.total_checks_per_cloud,
                        "total_redteam_vectors_per_cloud": report.total_redteam_vectors_per_cloud,
                    })
                );
                if report.all_passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(err) => {
                eprintln!("error: carbon coverage-audit failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::Engine { work_dir } => {
            let (Ok(runtime_api), is_aws_prod) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon engine`");
                return ExitCode::from(2);
            };
            run_engine_runtime(&runtime_api, work_dir, is_aws_prod)
        }
        #[cfg(feature = "eval")]
        Command::Eval { scratch_dir } => {
            let (Ok(runtime_api), _) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon eval`");
                return ExitCode::from(2);
            };
            run_eval_runtime(&runtime_api, scratch_dir)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_train_command(
    verb_label: &str,
    common: CommonTrainArgs,
    csv: Option<PathBuf>,
    target: Option<String>,
    synthetic_rows: usize,
    features: usize,
    stop_after_ms: Option<u64>,
    resume: bool,
    predict_rows: usize,
    save: Option<PathBuf>,
    shred: bool,
    persist_handle: bool,
    pretty_stdout: bool,
    force_progress: bool,
    global_quiet: bool,
) -> ExitCode {
    let json = common.json;
    let json_mode = !pretty_stdout;
    let delim_override = match parse_delimiter_arg(common.delimiter.as_deref()) {
        Ok(d) => d,
        Err(e) => return emit_cli_error(json, &e),
    };
    let explicit_tier =
        match resolve_explicit_tier_flag(common.free, common.pro, common.tier.as_deref()) {
            Ok(t) => t,
            Err(e) => return emit_cli_error(json, &e),
        };
    let Ok(effort_val) = parse_api_effort(&common.effort) else {
        return emit_cli_error(json, &format!("invalid --effort `{}`", common.effort));
    };
    let cloud_clone = common.cloud.clone();
    let (endpoints, pinned_keks, pre_probed, tenant_id, eff_tier, auth_token) = if common.dry_run {
        (
            Vec::new(),
            BTreeMap::new(),
            None,
            common.tenant.unwrap_or_else(|| "dry_run_tenant".to_owned()),
            explicit_tier.unwrap_or(CarbonTierWire::Free),
            None,
        )
    } else {
        match resolve_cli_target_and_auth(
            common.endpoint,
            common.cloud,
            common.clouds_manifest.as_deref(),
            common.environment.as_deref(),
            &common.region,
            explicit_tier,
            common.tenant,
            "cli_tenant",
            common.token,
            common.token_file,
            CarbonTierWire::Pro,
        ) {
            Ok(r) => (
                r.endpoints,
                r.pinned_keks,
                Some((r.probed_client, r.first_probe)),
                r.tenant_id,
                r.tier,
                r.auth_token,
            ),
            Err((code, msg)) => {
                if json {
                    return emit_cli_error(true, &msg);
                }
                eprintln!("error: {msg}");
                return code;
            }
        }
    };
    let target_was_explicit = target.is_some();
    let target_column = target.unwrap_or_else(|| {
        if common.no_header {
            "-1".to_owned()
        } else {
            "y".to_owned()
        }
    });
    let show_stderr_progress =
        !global_quiet && !pretty_stdout && (force_progress || std::io::stderr().is_terminal());
    let opts = CarbonCliRunOptions {
        endpoints,
        csv_path: csv,
        target_column,
        synthetic_rows,
        synthetic_features: features,
        tenant_id,
        tier: eff_tier,
        effort: effort_val,
        objective: common.objective,
        seed: common.seed,
        holdout_fraction: 0.20,
        stop_after_ms,
        resume_after_stop: resume,
        predict_rows,
        auth_token,
        shred_after_run: shred,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks,
        pre_probed,
        target_was_explicit,
        show_stderr_progress,
        delimiter_override: delim_override,
        no_header: common.no_header,
        persist_handle,
        save_model_path: if common.dry_run { None } else { save },
        cloud_name: cloud_clone,
        split_mode: common.split,
        calibration_mode: common.calibration,
        extra_na_values: common.na_values,
        inf_as_nan: common.inf_as_nan,
        embed_token: common.embed_token,
        dry_run: common.dry_run,
        allow_unencrypted_enterprise: common.allow_unencrypted_enterprise,
    };
    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => {
            if pretty_stdout {
                let _ = write_stdout_line(&render_pretty_run_scorecard(&receipt));
                ExitCode::SUCCESS
            } else {
                match serde_json::to_string_pretty(&receipt) {
                    Ok(s) => {
                        let _ = write_stdout_line(&s);
                        ExitCode::SUCCESS
                    }
                    Err(e) => emit_cli_error(json_mode, &format!("serialize receipt: {e}")),
                }
            }
        }
        Err(err) => emit_cli_error(json, &format!("carbon {verb_label} failed: {err}")),
    }
}

fn parse_delimiter_arg(raw: Option<&str>) -> Result<Option<char>, String> {
    let Some(s) = raw else {
        return Ok(None);
    };
    match s.trim() {
        "" | "auto" => Ok(None),
        "," | "comma" | "csv" => Ok(Some(',')),
        "\t" | "\\t" | "tab" | "tsv" => Ok(Some('\t')),
        ";" | "semicolon" | "semi" => Ok(Some(';')),
        "|" | "pipe" => Ok(Some('|')),
        other if other.chars().count() == 1 => Ok(other.chars().next()),
        other => Err(format!(
            "unsupported --delimiter `{other}` (expected `,`, `;`, `tab`, `|`, or `auto`)"
        )),
    }
}

fn should_emit_pretty_stdout(format_arg: &str, force_json: bool) -> bool {
    if force_json {
        return false;
    }
    match format_arg.trim().to_ascii_lowercase().as_str() {
        "pretty" | "human" | "table" => true,
        "json" => false,
        _ => std::io::stdout().is_terminal(),
    }
}

fn render_pretty_inspect_report(report: &CarbonTabularInspectReport) -> String {
    let mut out = String::with_capacity(1_536);
    let delim_name = match report.delimiter.as_str() {
        "\t" => "TAB (TSV)",
        "," => "',' (CSV)",
        ";" => "';' (Semicolon)",
        other => other,
    };
    let free_eligible = report.valid_target_rows <= 5_000 && report.encoded_feature_count <= 64;
    out.push_str("┌─ Carbon Local Dataset Preflight (`carbon inspect` — zero network) ────────┐\n");
    out.push_str(&format!("  File          : {}\n", report.file_path));
    out.push_str(&format!(
        "  Shape         : {} rows x {} raw columns ({} encoded features) | Delimiter = {}\n",
        report.raw_rows, report.raw_columns, report.encoded_feature_count, delim_name
    ));
    out.push_str(&format!(
        "  Target Column : '{}' (valid={} rows, min={:.4}, median={:.4}, max={:.4}, 3σ-outliers={})\n",
        report.target_column,
        report.valid_target_rows,
        report.target_min,
        report.target_median,
        report.target_max,
        report.target_outliers_3sigma
    ));
    out.push_str(&format!(
        "  Recommended   : --effort {} (Free eligible = {})\n\n",
        report.recommended_effort, free_eligible
    ));
    out.push_str(&format!(
        "  {:<22}  {:<18}  {:>9}  {:>8}  {}\n",
        "COLUMN", "DETECTED_KIND", "MISSING%", "UNIQUE", "ROLE"
    ));
    for c in report.columns.iter().take(24) {
        out.push_str(&format!(
            "  {:<22}  {:<18}  {:>8.1}%  {:>8}  {}\n",
            c.name, c.detected_type, c.missing_pct, c.distinct_count, c.role
        ));
    }
    if report.columns.len() > 24 {
        out.push_str(&format!(
            "  ... ({} additional columns; pass `--json` for full schema)\n",
            report.columns.len() - 24
        ));
    }
    if !report.effort_previews.is_empty() {
        out.push_str("\n  Estimated Serverless Execution & Cost Preview:\n");
        for p in &report.effort_previews {
            let mode_label = format!("{:?}", p.execution_mode);
            out.push_str(&format!(
                "    [{:<4}] --effort {:<6} : {:<16} | {:>2} wave(s), {:>2} shard(s) | ~{:>5.1}s | ~${:.6} USD\n",
                p.tier,
                p.effort,
                mode_label,
                p.planned_waves,
                p.eval_shards,
                p.estimated_wall_secs,
                p.estimated_usd
            ));
        }
    }
    if !report.warnings.is_empty() {
        out.push_str("\n  Notes:\n");
        for w in &report.warnings {
            out.push_str(&format!("    - {w}\n"));
        }
    }
    out.push_str(&format!(
        "\n  Ready: carbon fit \"{}\" --target \"{}\" --save model.carbon.json\n",
        report.file_path, report.target_column
    ));
    out.push_str("└───────────────────────────────────────────────────────────────────────────┘");
    out
}

fn parse_export_lang(raw: &str) -> Result<CarbonExportLangWire, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "python" | "py" => Ok(CarbonExportLangWire::Python),
        "rust" | "rs" => Ok(CarbonExportLangWire::Rust),
        "rust-refittable" | "rust_refittable" => Ok(CarbonExportLangWire::RustRefittable),
        "js" | "javascript" | "ts" => Ok(CarbonExportLangWire::Js),
        "c" => Ok(CarbonExportLangWire::C),
        "cpp" | "c++" | "cxx" => Ok(CarbonExportLangWire::Cpp),
        other => Err(format!(
            "unsupported --lang `{other}` (expected `python`, `rust`, `rust-refittable`, `js`, `c`, or `cpp`)"
        )),
    }
}

fn render_pretty_explain_response(resp: &CarbonExplainResponse) -> String {
    let mut out = String::with_capacity(1_024);
    let cuen = resp.encrypted_explain_cuen_b64.is_some();
    out.push_str("┌─ Carbon Zero-Knowledge Champion Explanation (`carbon explain`) ───────────┐\n");
    out.push_str(&format!(
        "  Run & Method  : {} | {} (CUEN\\x01={cuen})\n",
        resp.run_id, resp.method
    ));
    out.push_str(&format!(
        "  Baseline Loss : {:.6}\n\n",
        resp.baseline_holdout_loss
    ));
    out.push_str(&format!(
        "  {:<26}  {:>12}  {:>14}\n",
        "FEATURE", "IMPORTANCE%", "DELTA_LOSS"
    ));
    for f in &resp.feature_importances {
        out.push_str(&format!(
            "  {:<26}  {:>11.2}%  {:>+14.6}\n",
            f.feature,
            f.importance * 100.0,
            f.delta_loss
        ));
    }
    out.push_str("└───────────────────────────────────────────────────────────────────────────┘");
    out
}

fn execute_fitness_command(action: FitnessCommand) -> ExitCode {
    match action {
        FitnessCommand::List { json } => {
            let catalog = ObjectiveSpec::catalog();
            if !json && std::io::stdout().is_terminal() {
                let mut out = String::with_capacity(3_072);
                out.push_str(&format!(
                    "{:<24}  {:<9}  {:<6}  {}\n",
                    "OBJECTIVE", "DIRECTION", "ORDER?", "DESCRIPTION"
                ));
                for entry in &catalog {
                    out.push_str(&format!(
                        "{:<24}  {:<9}  {:<6}  {}\n",
                        entry.name,
                        entry.direction.as_str(),
                        if entry.requires_order { "yes" } else { "no" },
                        entry.description
                    ));
                }
                let _ = write_stdout_line(out.trim_end_matches('\n'));
            } else if let Ok(s) = serde_json::to_string_pretty(&catalog) {
                let _ = write_stdout_line(&s);
            }
            ExitCode::SUCCESS
        }
        FitnessCommand::Validate { spec, json } => match resolve_objective_argument(&spec) {
            Ok((wire, parsed)) => {
                let aux: Vec<String> = parsed.columns().into_iter().collect();
                let res: Vec<String> = parsed.resources().into_iter().collect();
                let payload = serde_json::json!({
                    "valid": true,
                    "name": parsed.name(),
                    "direction": parsed.direction(),
                    "requires_order": parsed.requires_order(),
                    "uses_execution_telemetry": parsed.uses_execution_telemetry(),
                    "benefits_from_post_hoc_calibration": parsed.benefits_from_post_hoc_calibration(),
                    "aux_columns": aux,
                    "resources": res,
                    "fitness_sha256": parsed.identity_sha256(),
                    "wire_objective": wire,
                    "canonical_spec": parsed.as_value(),
                });
                if let Ok(s) = serde_json::to_string_pretty(&payload) {
                    let _ = write_stdout_line(&s);
                }
                ExitCode::SUCCESS
            }
            Err(err) => emit_cli_error(json, &err),
        },
        FitnessCommand::Schema { json: _ } => {
            if let Ok(s) = serde_json::to_string_pretty(&ObjectiveSpec::schema_json()) {
                let _ = write_stdout_line(&s);
            }
            ExitCode::SUCCESS
        }
        FitnessCommand::Evaluate {
            spec,
            csv,
            target,
            pred,
            json,
        } => {
            let (_, parsed_spec) = match resolve_objective_argument(&spec) {
                Ok(v) => v,
                Err(e) => return emit_cli_error(json, &e),
            };
            let table = match load_delimited_table(&csv, None, true) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let find_col = |name: &str| -> Result<usize, String> {
                table.headers.iter().position(|h| h == name).ok_or_else(|| {
                    format!(
                        "column '{name}' not found in {} (headers: [{}])",
                        csv.display(),
                        table.headers.join(", ")
                    )
                })
            };
            let t_idx = match find_col(&target) {
                Ok(i) => i,
                Err(e) => return emit_cli_error(json, &e),
            };
            let p_idx = match find_col(&pred) {
                Ok(i) => i,
                Err(e) => return emit_cli_error(json, &e),
            };
            let mut aux_indices = Vec::new();
            for aux_name in parsed_spec.columns() {
                match find_col(&aux_name) {
                    Ok(i) => aux_indices.push((aux_name, i)),
                    Err(e) => return emit_cli_error(json, &e),
                }
            }
            let mut y_vec = Vec::with_capacity(table.rows.len());
            let mut p_vec = Vec::with_capacity(table.rows.len());
            let mut aux_map: BTreeMap<String, Vec<f64>> = aux_indices
                .iter()
                .map(|(n, _)| (n.clone(), Vec::with_capacity(table.rows.len())))
                .collect();
            for (r_idx, row) in table.rows.iter().enumerate() {
                let parse_cell = |col_i: usize, label: &str| -> Result<f64, String> {
                    let raw = row.get(col_i).and_then(|c| c.as_deref()).unwrap_or("");
                    parse_flexible_numeric_f64(raw)
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| {
                            format!(
                                "row {} column '{label}' ('{raw}') is not a finite number",
                                r_idx + 2
                            )
                        })
                };
                let yv = match parse_cell(t_idx, &target) {
                    Ok(v) => v,
                    Err(e) => return emit_cli_error(json, &e),
                };
                let pv = match parse_cell(p_idx, &pred) {
                    Ok(v) => v,
                    Err(e) => return emit_cli_error(json, &e),
                };
                for (aux_name, aux_i) in &aux_indices {
                    let av = match parse_cell(*aux_i, aux_name) {
                        Ok(v) => v,
                        Err(e) => return emit_cli_error(json, &e),
                    };
                    if let Some(slot) = aux_map.get_mut(aux_name) {
                        slot.push(av);
                    }
                }
                y_vec.push(yv);
                p_vec.push(pv);
            }
            match parsed_spec.evaluate(&y_vec, &p_vec, &aux_map) {
                Ok(score) => {
                    let payload = serde_json::json!({
                        "rows_scored": y_vec.len(),
                        "objective": parsed_spec.name(),
                        "value": score.value,
                        "loss": score.loss,
                        "direction": score.direction,
                        "fitness_sha256": score.fitness_sha256,
                    });
                    if let Ok(s) = serde_json::to_string_pretty(&payload) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => emit_cli_error(json, &e),
            }
        }
    }
}

fn emit_shell_completions(shell: &str) -> Result<String, String> {
    let subcommands = [
        "version",
        "self-verify",
        "register",
        "login",
        "whoami",
        "logout",
        "doctor",
        "ping",
        "inspect",
        "fit",
        "predict",
        "explain",
        "export",
        "resume",
        "fitness",
        "runs",
        "status",
        "shred",
        "upgrade",
        "completions",
        "quickstart",
        "run",
        "bench",
        "redteam",
        "token-mint",
        "plan",
        "spot-check",
    ];
    match shell.trim().to_ascii_lowercase().as_str() {
        "bash" => Ok(format!(
            "# bash completion for carbon\n_carbon_completions() {{\n    local cur prev cmds\n    COMPREPLY=()\n    cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n    prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"\n    cmds=\"{cmds}\"\n    if [[ ${{COMP_CWORD}} -eq 1 ]]; then\n        COMPREPLY=( $(compgen -W \"${{cmds}} --help --version\" -- \"${{cur}}\") )\n        return 0\n    fi\n    case \"${{prev}}\" in\n        --effort) COMPREPLY=( $(compgen -W \"fast medium high ultra\" -- \"${{cur}}\") ); return 0 ;;\n        --cloud) COMPREPLY=( $(compgen -W \"prod dev\" -- \"${{cur}}\") ); return 0 ;;\n        --format) COMPREPLY=( $(compgen -W \"auto pretty json\" -- \"${{cur}}\") ); return 0 ;;\n        --suite) COMPREPLY=( $(compgen -W \"ops smoke standard comprehensive\" -- \"${{cur}}\") ); return 0 ;;\n        completions) COMPREPLY=( $(compgen -W \"bash zsh fish powershell\" -- \"${{cur}}\") ); return 0 ;;\n    esac\n    COMPREPLY=( $(compgen -f -- \"${{cur}}\") )\n}}\ncomplete -F _carbon_completions carbon\n",
            cmds = subcommands.join(" ")
        )),
        "zsh" => Ok(format!(
            "#compdef carbon\n_carbon() {{\n    local -a subcmds\n    subcmds=({cmds})\n    if (( CURRENT == 2 )); then\n        _describe 'carbon command' subcmds\n        _files -g '*.csv *.tsv'\n    else\n        _arguments \\\n            '--target=[Target column]:column:' \\\n            '--effort=[Search effort]:effort:(fast medium high ultra)' \\\n            '--cloud=[Public cloud]:cloud:(prod dev)' \\\n            '--format=[Output format]:format:(auto pretty json)' \\\n            '--suite=[Benchmark suite]:suite:(ops smoke standard comprehensive)' \\\n            '--save=[Model handle output]:file:_files' \\\n            '--model=[Saved model handle]:file:_files' \\\n            '--out=[Predictions CSV]:file:_files' \\\n            '*:file:_files'\n    fi\n}}\n_carbon \"$@\"\n",
            cmds = subcommands.join(" ")
        )),
        "fish" => {
            let mut s = String::from("# fish completion for carbon\n");
            for cmd in subcommands {
                s.push_str(&format!(
                    "complete -c carbon -n '__fish_use_subcommand' -a '{cmd}'\n"
                ));
            }
            s.push_str("complete -c carbon -l effort -xa 'fast medium high ultra'\n");
            s.push_str("complete -c carbon -l cloud -xa 'prod dev'\n");
            s.push_str("complete -c carbon -l format -xa 'auto pretty json'\n");
            s.push_str("complete -c carbon -l suite -xa 'ops smoke standard comprehensive'\n");
            Ok(s)
        }
        "powershell" | "pwsh" => Ok(format!(
            "# PowerShell completion for carbon\nRegister-ArgumentCompleter -Native -CommandName carbon -ScriptBlock {{\n    param($wordToComplete, $commandAst, $cursorPosition)\n    $cmds = @({quoted})\n    $cmds | Where-Object {{ $_ -like \"$wordToComplete*\" }} | ForEach-Object {{\n        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)\n    }}\n}}\n",
            quoted = subcommands
                .iter()
                .map(|c| format!("'{c}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        other => Err(format!(
            "unsupported shell `{other}` (expected bash, zsh, fish, or powershell)"
        )),
    }
}
