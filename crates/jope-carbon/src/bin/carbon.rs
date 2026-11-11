//! Multi-role `carbon` CLI and AWS Lambda custom-runtime entrypoint (`package`, `deploy`,
//! `rollback`, `run`, `bench`, `redteam`, `token-mint`, `serve`, `emu`, `engine`, `eval`,
//! `plan`, and `spot-check`).

use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use jope_api::{carbon::CarbonTierWire, Effort};
use jope_carbon::{
    evaluate_spot_hybrid_gates, execute_carbon_cli_benchmark_with_token, execute_carbon_cli_run,
    execute_carbon_deploy, execute_carbon_package, execute_carbon_redteam_suite,
    execute_carbon_rollback, mint_capability_token_from_vault, now_epoch_ms, open_eval_invoker,
    open_remote_cas, plan_carbon_execution, render_benchmark_markdown, render_redteam_markdown,
    resolve_cell_stack_endpoint, run_lambda_custom_runtime_loop, CarbonAuthVerifier,
    CarbonCapabilityClaims, CarbonCliRunOptions, CarbonDeployOptions, CarbonEffortTier,
    CarbonEmuConfig, CarbonEngineKek, CarbonEngineWorker, CarbonEnterpriseServeConfig,
    CarbonEnterpriseServer, CarbonEvalWorker, CarbonLiveAliasVersions, CarbonLocalLambdaEmulator,
    CarbonPackageOptions, CarbonPlanInput, CarbonProductTier, CarbonRole, CarbonRollbackOptions,
    CarbonSealKey, CarbonThinClient, MonthlyWindowMetrics, SpotAdvisorThresholds,
};

#[derive(Debug, Parser)]
#[command(
    name = "carbon",
    version,
    about = "Carbon zero-knowledge serverless CLI client, benchmark & red-team harness, Lambda runtime, Enterprise server, and CI emulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print build, role, and feature readiness as JSON.
    Version,
    /// Cryptographically verify `r882`, cross-compile ARM64 `carbon-edge` (`CT3` isolated) + `carbon`,
    /// and emit deterministic reproducible `.zip` bundles + `release-manifest.json`.
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
    },
    /// Deploy a packaged Carbon cell via AWS CloudFormation ChangeSet + Secrets Manager + `:live` aliases,
    /// followed by mandatory `/ping` and 16-vector live `carbon redteam` verification.
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
    Rollback {
        /// Target environment (`dev`, `staging`, or `prod`).
        #[arg(long = "env", default_value = "dev")]
        environment: String,
        /// Target AWS region.
        #[arg(long, default_value = "us-east-2")]
        region: String,
        /// Optional explicit `<edge_ver>,<engine_ver>,<eval_ver>` override.
        #[arg(long, value_delimiter = ',')]
        to_versions: Vec<String>,
    },
    /// Condition & encrypt a local CSV (or synthetic matrix), upload via size-pinned presigned URL,
    /// run `r882`-guided GP waves, and execute E2E encrypted prediction.
    Run {
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Optional stack environment (`dev`, `staging`, `prod`) to auto-resolve endpoint & vault token.
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
        #[arg(long)]
        csv: Option<PathBuf>,
        #[arg(long, default_value = "y")]
        target: String,
        #[arg(long, default_value_t = 240)]
        synthetic_rows: usize,
        #[arg(long, default_value_t = 6)]
        features: usize,
        #[arg(long, default_value = "cli_tenant")]
        tenant: String,
        #[arg(long, default_value = "pro")]
        tier: String,
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
    },
    /// Run the Free + Pro `r882` zero-knowledge live Carbon Lambda / Enterprise server benchmark suite.
    Bench {
        #[arg(long, value_delimiter = ',')]
        endpoint: Vec<String>,
        /// Optional stack environment (`dev`, `staging`, `prod`) to auto-resolve endpoint & vault token.
        #[arg(long = "env")]
        environment: Option<String>,
        #[arg(long, default_value = "us-east-2")]
        region: String,
        #[arg(long, default_value = "pro")]
        tier: String,
        #[arg(long, default_value = "bench_tenant")]
        tenant: String,
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
    Serve {
        #[arg(long, default_value = "127.0.0.1:8088")]
        listen: SocketAddr,
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "vpc-local-1")]
        region: String,
        #[arg(long)]
        enable_user_envelope: bool,
    },
    /// Boot the local multi-cell Lambda Runtime API + S3 CAS emulator (`--self-test` for CI).
    Emu {
        #[arg(long)]
        work_dir: PathBuf,
        #[arg(long)]
        self_test: bool,
    },
    /// Run the sealed `carbon engine` AWS Lambda Custom Runtime (`2018-06-01`) worker loop.
    Engine {
        #[arg(long)]
        work_dir: PathBuf,
    },
    /// Run the stateless `carbon eval` AWS Lambda Custom Runtime (`2018-06-01`) shard worker loop.
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

fn decode_32byte_hex_or_default(env_key: &str, fallback: [u8; 32]) -> Result<[u8; 32], String> {
    match env::var(env_key) {
        Ok(val) if !val.trim().is_empty() => decode_required_32byte_hex(env_key),
        _ => Ok(fallback),
    }
}

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
    let seal_secret = if is_aws_prod {
        match decode_required_32byte_hex("CARBON_SEAL_SECRET_HEX") {
            Ok(s) => s,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(2);
            }
        }
    } else {
        match decode_32byte_hex_or_default("CARBON_SEAL_SECRET_HEX", [0x77u8; 32]) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(2);
            }
        }
    };
    let engine_kek = if is_aws_prod {
        let Ok(kek_hex) = env::var("CARBON_ENGINE_KEK_HEX") else {
            eprintln!("error: missing required CARBON_ENGINE_KEK_HEX on AWS Lambda");
            return ExitCode::from(2);
        };
        match CarbonEngineKek::from_hex(&kek_hex) {
            Ok(k) => Some(Arc::new(k)),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(2);
            }
        }
    } else if let Ok(kek_hex) = env::var("CARBON_ENGINE_KEK_HEX") {
        if !kek_hex.trim().is_empty() {
            match CarbonEngineKek::from_hex(&kek_hex) {
                Ok(k) => Some(Arc::new(k)),
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            }
        } else {
            None
        }
    } else {
        None
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
    let worker = match CarbonEngineWorker::new(
        cas,
        eval_invoker,
        CarbonSealKey::from_bytes(seal_secret),
        work_dir,
    ) {
        Ok(w) => w
            .with_engine_kek(engine_kek)
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
                    "version": env!("CARGO_PKG_VERSION"),
                    "roles": [
                        CarbonRole::EngineOrchestrate.as_str(),
                        CarbonRole::EvalBatch.as_str(),
                        CarbonRole::EnterpriseServe.as_str(),
                        CarbonRole::LocalEmulator.as_str(),
                    ],
                })
            );
            ExitCode::SUCCESS
        }
        Command::Package {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            jobs,
        } => match execute_carbon_package(&CarbonPackageOptions {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            build_jobs: jobs,
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
        Command::Rollback {
            environment,
            region,
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
            match execute_carbon_rollback(&CarbonRollbackOptions {
                environment,
                region,
                target_versions,
            }) {
                Ok(receipt) => {
                    if let Ok(s) = serde_json::to_string_pretty(&receipt) {
                        println!("{s}");
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon rollback failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Run {
            mut endpoint,
            environment,
            region,
            csv,
            target,
            synthetic_rows,
            features,
            tenant,
            tier,
            effort,
            objective,
            seed,
            stop_after_ms,
            resume,
            predict_rows,
            shred,
        } => {
            let (Ok(tier_wire), Ok(effort_val)) =
                (parse_tier_wire(&tier), parse_api_effort(&effort))
            else {
                eprintln!("error: invalid --tier or --effort");
                return ExitCode::from(2);
            };
            let mut auth_token = None;
            if let Some(env_name) = &environment {
                if endpoint.is_empty() {
                    match resolve_cell_stack_endpoint(env_name, &region) {
                        Ok(url) => endpoint.push(url),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                if tier_wire != CarbonTierWire::Free
                    && env::var("CARBON_AUTH_TOKEN").is_err()
                    && env::var("CARBON_AUTH_SECRET_HEX").is_err()
                {
                    match mint_capability_token_from_vault(
                        env_name, &region, &tenant, tier_wire, 3_600, false,
                    ) {
                        Ok(tok) => auth_token = Some(tok),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if endpoint.is_empty() {
                eprintln!("error: pass --endpoint <URL> or --env <dev|staging|prod>");
                return ExitCode::from(2);
            }
            let opts = CarbonCliRunOptions {
                endpoints: endpoint,
                csv_path: csv,
                target_column: target,
                synthetic_rows,
                synthetic_features: features,
                tenant_id: tenant,
                tier: tier_wire,
                effort: effort_val,
                objective,
                seed,
                holdout_fraction: 0.20,
                stop_after_ms,
                resume_after_stop: resume,
                predict_rows,
                auth_token,
                shred_after_run: shred,
            };
            match execute_carbon_cli_run(&opts) {
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
            mut endpoint,
            environment,
            region,
            tier,
            tenant,
            json_out,
            markdown_out,
        } => {
            let Ok(tier_wire) = parse_tier_wire(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            let mut pro_token = None;
            if let Some(env_name) = &environment {
                if endpoint.is_empty() {
                    match resolve_cell_stack_endpoint(env_name, &region) {
                        Ok(url) => endpoint.push(url),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                if tier_wire != CarbonTierWire::Free
                    && env::var("CARBON_AUTH_TOKEN").is_err()
                    && env::var("CARBON_AUTH_SECRET_HEX").is_err()
                {
                    match mint_capability_token_from_vault(
                        env_name, &region, &tenant, tier_wire, 3_600, false,
                    ) {
                        Ok(tok) => pro_token = Some(tok),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if endpoint.is_empty() {
                eprintln!("error: pass --endpoint <URL> or --env <dev|staging|prod>");
                return ExitCode::from(2);
            }
            match execute_carbon_cli_benchmark_with_token(endpoint, tier_wire, &tenant, pro_token) {
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
            environment,
            region,
            json_out,
            markdown_out,
        } => {
            let resolved_ep = match (endpoint, environment) {
                (Some(ep), _) => ep,
                (None, Some(env_name)) => match resolve_cell_stack_endpoint(&env_name, &region) {
                    Ok(url) => url,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                },
                (None, None) => {
                    eprintln!("error: pass --endpoint <URL> or --env <dev|staging|prod>");
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
            let Ok(tier_wire) = parse_tier_wire(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            if let Some(env_name) = environment {
                return match mint_capability_token_from_vault(
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
        Command::Serve {
            listen,
            data_dir,
            region,
            enable_user_envelope,
        } => {
            let mut cfg = CarbonEnterpriseServeConfig::enterprise_default(listen, data_dir);
            cfg.region = region;
            cfg.enable_user_envelope = enable_user_envelope;
            cfg.seal_secret =
                match decode_32byte_hex_or_default("CARBON_SEAL_SECRET_HEX", cfg.seal_secret) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                };
            cfg.presign_secret =
                match decode_32byte_hex_or_default("CARBON_PRESIGN_SECRET_HEX", cfg.presign_secret)
                {
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
        Command::Engine { work_dir } => {
            let (Ok(runtime_api), is_aws_prod) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon engine`");
                return ExitCode::from(2);
            };
            run_engine_runtime(&runtime_api, work_dir, is_aws_prod)
        }
        Command::Eval { scratch_dir } => {
            let (Ok(runtime_api), _) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon eval`");
                return ExitCode::from(2);
            };
            run_eval_runtime(&runtime_api, scratch_dir)
        }
    }
}
