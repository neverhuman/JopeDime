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
use clap::{Parser, Subcommand};
use jope_api::{carbon::CarbonTierWire, Effort};
#[cfg(any(
    feature = "engine",
    feature = "eval",
    feature = "serve",
    feature = "emu"
))]
use jope_carbon::CarbonRole;
use jope_carbon::{
    decode_p256_pubkey_b64, evaluate_spot_hybrid_gates, execute_carbon_cli_benchmark_with_token,
    execute_carbon_cli_run_with_ux, execute_carbon_redteam_suite,
    generate_nonlinear_benchmark_rows, list_stored_cloud_tokens, load_stored_cloud_token,
    now_epoch_ms, plan_carbon_execution, remove_stored_cloud_token, render_benchmark_markdown,
    render_redteam_markdown, resolve_public_cloud_endpoint, save_stored_cloud_token,
    sha256_lower_hex, verified_absolute_jopedime_config_dir, verify_ecdsa_p256_spki_pem_signature,
    CarbonAuthVerifier, CarbonCapabilityClaims, CarbonCliRunOptions, CarbonCliRunUxConfig,
    CarbonEffortTier, CarbonPlanInput, CarbonProbedEndpoint, CarbonProductTier,
    CarbonPublicCloudsManifest, CarbonThinClient, MonthlyWindowMetrics, SpotAdvisorThresholds,
    CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET, CARBON_GIT_SHA, CARBON_PKG_VERSION,
    CARBON_RELEASE_TAG, DEFAULT_PUBLIC_CLOUDS_URL, FALLBACK_PUBLIC_CLOUDS_URL,
    MAX_MINT_TOKEN_TTL_SECS,
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

const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
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

#[derive(Debug, Parser)]
#[command(
    name = "carbon",
    version = LONG_VERSION,
    args_conflicts_with_subcommands = true,
    about = "Carbon zero-knowledge serverless CLI client, benchmark & red-team harness, Lambda runtime, Enterprise server, and CI emulator"
)]
struct Cli {
    /// Positional shorthand (`carbon data.csv`): conditions & encrypts the CSV client-side,
    /// runs a `fast` zero-knowledge cloud job, decrypts predictions, and shreds remote blobs.
    #[arg(value_name = "CSV_FILE")]
    csv_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print build, git SHA, release tag, target triple, and compiled role readiness as JSON.
    Version,
    /// Cryptographically verify the installed `carbon` binary SHA-256, `carbon-clouds.json` P-256 KEK pins,
    /// embedded release public key fingerprint, and ECDSA-P256 `SHA256SUMS.txt.sig` signature.
    SelfVerify {
        /// Optional path to `SHA256SUMS.txt` (defaults to `$CARBON_CONFIG_DIR/SHA256SUMS.txt` if present).
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Optional path to base64-encoded ECDSA-P256 `SHA256SUMS.txt.sig` (defaults to `<manifest>.sig`).
        #[arg(long)]
        signature: Option<PathBuf>,
        /// Optional explicit `carbon-clouds.json` path override.
        #[arg(long)]
        clouds_manifest: Option<PathBuf>,
        /// Optional PEM public key override for offline CI/E2E fixture testing only.
        #[arg(long)]
        allow_test_pubkey: Option<PathBuf>,
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
        #[arg(long)]
        csv: Option<PathBuf>,
        /// Target column name (defaults to 'y', or auto-selects 'target' when 'y' is absent).
        #[arg(long)]
        target: Option<String>,
        #[arg(long, default_value_t = 240)]
        synthetic_rows: usize,
        #[arg(long, default_value_t = 6)]
        features: usize,
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
        #[arg(long, default_value = "rmse")]
        objective: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        stop_after_ms: Option<u64>,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 5)]
        predict_rows: usize,
        /// Cryptographically shred (`DELETE /v1/carbon/runs/{id}`) blobs and state upon completion.
        #[arg(long)]
        shred: bool,
        /// Force live wave progress and human summary card on stderr even when stderr is not a TTY.
        #[arg(long, conflicts_with = "quiet")]
        progress: bool,
        /// Suppress interactive stderr progress and summary card (stdout remains pure JSON).
        #[arg(long, conflicts_with = "progress")]
        quiet: bool,
    },
    /// Run the Free + Pro `r882` zero-knowledge live Carbon Lambda / Enterprise server benchmark suite.
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
        #[arg(long, default_value_t = 35.0)]
        ms_per_candidate: f64,
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
        /// Suppress interactive step-by-step walkthrough lines on stderr.
        #[arg(long)]
        quiet: bool,
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
        let lookup_cloud = cloud.as_deref().unwrap_or("prod");
        if cloud.is_some() || !first_probe.enterprise_self_hosted {
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

    let embedded_pub_sha = sha256_lower_hex(EMBEDDED_RELEASE_SIGNING_PUB_PEM.as_bytes());
    if embedded_pub_sha != EXPECTED_SIGNING_PUB_SHA256 {
        eprintln!(
            "error: embedded release public key SHA-256 mismatch (got {embedded_pub_sha}, expected {EXPECTED_SIGNING_PUB_SHA256})"
        );
        return ExitCode::from(1);
    }

    let (pubkey_pem, using_test_pubkey) = if let Some(test_pub) = &allow_test_pubkey {
        match fs::read_to_string(test_pub) {
            Ok(pem) => (pem, true),
            Err(e) => {
                eprintln!(
                    "error: read --allow-test-pubkey {}: {e}",
                    test_pub.display()
                );
                return ExitCode::from(2);
            }
        }
    } else {
        (EMBEDDED_RELEASE_SIGNING_PUB_PEM.to_owned(), false)
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
    let effective_manifest_path = manifest_arg.or_else(|| {
        config_dir_opt
            .as_ref()
            .map(|d| d.join("SHA256SUMS.txt"))
            .filter(|p| p.is_file())
    });

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
    } else if explicit_manifest {
        eprintln!("error: --manifest file not found");
        return ExitCode::from(1);
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
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

fn execute_positional_csv_shorthand(csv_path: PathBuf) -> ExitCode {
    let looks_like_tabular_path = csv_path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv") || ext.eq_ignore_ascii_case("tsv"))
        || csv_path.to_string_lossy().contains('/')
        || csv_path.to_string_lossy().contains('\\');

    if !csv_path.is_file() {
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
    let show_stderr_progress = std::io::stderr().is_terminal();
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
                let mut dir_mode_ok = true;
                let mut dir_mode_octal = None;
                if dir.exists() {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        if let Ok(meta) = fs::metadata(&dir) {
                            let m = meta.permissions().mode() & 0o777;
                            dir_mode_octal = Some(format!("{m:04o}"));
                            if (m & 0o077) != 0 {
                                dir_mode_ok = false;
                            }
                        }
                    }
                }
                let cred_path = dir.join("carbon-credentials.json");
                let mut cred_ok = true;
                let mut cred_mode_octal = None;
                if cred_path.is_file() {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        if let Ok(meta) = fs::metadata(&cred_path) {
                            let m = meta.permissions().mode() & 0o777;
                            cred_mode_octal = Some(format!("{m:04o}"));
                            cred_ok = m == 0o600;
                        }
                    }
                }
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
            env!("CARGO_PKG_VERSION"),
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
    let cli = Cli::parse();
    if let Some(csv_path) = cli.csv_file {
        return execute_positional_csv_shorthand(csv_path);
    }
    #[cfg(feature = "engine")]
    if cli.command.is_none() {
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
        } => execute_self_verify_command(manifest, signature, clouds_manifest, allow_test_pubkey),
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
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&serde_json::json!({
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
                            .unwrap_or_default()
                        );
                        ExitCode::SUCCESS
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
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
            csv,
            target,
            synthetic_rows,
            features,
            tenant,
            free,
            pro,
            tier,
            token,
            token_file,
            effort,
            objective,
            seed,
            stop_after_ms,
            resume,
            predict_rows,
            shred,
            progress,
            quiet,
        } => {
            let explicit_tier = match resolve_explicit_tier_flag(free, pro, tier.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let Ok(effort_val) = parse_api_effort(&effort) else {
                eprintln!("error: invalid --effort `{effort}`");
                return ExitCode::from(2);
            };
            let resolved = match resolve_cli_target_and_auth(
                endpoint,
                cloud,
                clouds_manifest.as_deref(),
                environment.as_deref(),
                &region,
                explicit_tier,
                tenant,
                "cli_tenant",
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
            let target_was_explicit = target.is_some();
            let target_column = target.unwrap_or_else(|| "y".to_owned());
            let show_stderr_progress = !quiet && (progress || std::io::stderr().is_terminal());
            let opts = CarbonCliRunOptions {
                endpoints: resolved.endpoints,
                csv_path: csv,
                target_column,
                synthetic_rows,
                synthetic_features: features,
                tenant_id: resolved.tenant_id,
                tier: resolved.tier,
                effort: effort_val,
                objective,
                seed,
                holdout_fraction: 0.20,
                stop_after_ms,
                resume_after_stop: resume,
                predict_rows,
                auth_token: resolved.auth_token,
                shred_after_run: shred,
            };
            let ux = CarbonCliRunUxConfig {
                pinned_keks: resolved.pinned_keks,
                pre_probed: Some((resolved.probed_client, resolved.first_probe)),
                target_was_explicit,
                show_stderr_progress,
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
                    eprintln!("error: carbon run failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Bench {
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
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
            match plan_carbon_execution(&CarbonPlanInput {
                product_tier,
                effort: effort_tier,
                total_rows,
                fit_rows,
                features,
                est_ms_per_candidate_1vcpu: ms_per_candidate,
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
                eprintln!("error: pass --self-test for automated emulator verification");
                ExitCode::from(2)
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
            quiet,
        } => run_quickstart_command(
            emit_csv,
            csv_only,
            cloud,
            clouds_manifest,
            endpoint,
            &effort,
            no_shred,
            quiet,
        ),
        #[cfg(feature = "emu")]
        Command::SpecReport {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
            source_sha,
            repo_root,
        } => match execute_carbon_spec_report(&CarbonSpecReportOptions {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
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
