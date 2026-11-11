//! Multi-role `carbon` CLI and Lambda custom-runtime entrypoint (`engine`, `eval`, `serve`, `emu`,
//! `plan`, and `spot-check`).

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use jope_api::{carbon::CarbonTierWire, Effort};
use jope_carbon::{
    evaluate_spot_hybrid_gates, plan_carbon_execution, run_lambda_custom_runtime_loop,
    CarbonEffortTier, CarbonEmuConfig, CarbonEngineWorker, CarbonEnterpriseServeConfig,
    CarbonEnterpriseServer, CarbonEvalWorker, CarbonHttpCas, CarbonHttpEvalInvoker,
    CarbonLocalLambdaEmulator, CarbonPlanInput, CarbonProductTier, CarbonRole, CarbonSealKey,
    CarbonThinClient, MonthlyWindowMetrics, SpotAdvisorThresholds,
};

#[derive(Debug, Parser)]
#[command(
    name = "carbon",
    version,
    about = "Carbon serverless cell runtime, adaptive planner, Enterprise server, and CI emulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print build, role, and feature readiness as JSON.
    Version,
    /// Compute the deterministic Solo vs Hybrid Fan-Out execution plan for a dataset shape.
    Plan {
        /// Product tier (`free`, `pro`, `enterprise`).
        #[arg(long, default_value = "pro")]
        tier: String,
        /// Effort budget (`fast`, `medium`, `high`, `ultra`).
        #[arg(long, default_value = "high")]
        effort: String,
        /// Total dataset row count (`FIT + HOLDOUT`).
        #[arg(long)]
        total_rows: usize,
        /// FIT row count (`< total_rows`).
        #[arg(long)]
        fit_rows: usize,
        /// Feature column count.
        #[arg(long)]
        features: usize,
        /// Estimated single-vCPU milliseconds per candidate (`default: 35.0`).
        #[arg(long, default_value_t = 35.0)]
        ms_per_candidate: f64,
    },
    /// Evaluate the 3-Gate Spot-Hybrid break-even advisor on a regional monthly snapshot.
    SpotCheck {
        /// AWS region name (for example `us-east-2`).
        #[arg(long, default_value = "us-east-2")]
        region: String,
        /// Monthly Lambda compute spend in USD (`Gate S-1`).
        #[arg(long)]
        monthly_spend_usd: f64,
        /// Sustained P20 active vCPUs (`Gate S-2`).
        #[arg(long)]
        p20_vcpus: f64,
        /// Monthly mean active vCPUs.
        #[arg(long)]
        mean_vcpus: f64,
        /// Monthly P95 active vCPUs (`Gate S-3`).
        #[arg(long)]
        p95_vcpus: f64,
        /// Observed cold-start rate (`0.0..=1.0`).
        #[arg(long, default_value_t = 0.01)]
        cold_start_rate: f64,
        /// Number of consecutive months at these metrics (`default: 1`).
        #[arg(long, default_value_t = 1)]
        consecutive_months: usize,
    },
    /// Run the self-hosted VPC / on-premises Enterprise single-binary server (`no_limit()`).
    Serve {
        /// TCP listen address (`host:port`).
        #[arg(long, default_value = "127.0.0.1:8088")]
        listen: SocketAddr,
        /// Persistent local CAS and encrypted checkpoint directory.
        #[arg(long)]
        data_dir: PathBuf,
        /// Logical region identifier advertised by `/v1/carbon/ping`.
        #[arg(long, default_value = "vpc-local-1")]
        region: String,
    },
    /// Boot the local multi-cell Lambda Runtime API + S3 CAS emulator (`--self-test` for CI).
    Emu {
        /// Scratch root directory for the emulated S3 CAS and containers.
        #[arg(long)]
        work_dir: PathBuf,
        /// Run a complete end-to-end multi-cell (`edge` + `engine` + `eval`) verification and exit.
        #[arg(long)]
        self_test: bool,
    },
    /// Run the sealed `carbon engine` AWS Lambda Custom Runtime (`2018-06-01`) worker loop.
    Engine {
        /// Working directory for staged `fit.jmat` and checkpoint packing.
        #[arg(long)]
        work_dir: PathBuf,
    },
    /// Run the stateless `carbon eval` AWS Lambda Custom Runtime (`2018-06-01`) shard worker loop.
    Eval {
        /// Scratch directory for warm-container `fit.jmat` staging.
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

fn decode_32byte_hex_or_default(env_key: &str, fallback: [u8; 32]) -> [u8; 32] {
    if let Ok(val) = env::var(env_key) {
        if let Ok(raw) = hex::decode(val.trim()) {
            if raw.len() == 32 {
                let mut out = [0u8; 32];
                out.copy_from_slice(&raw);
                return out;
            }
        }
    }
    fallback
}

fn run_emu_self_test(work_dir: PathBuf) -> Result<(), String> {
    std::fs::create_dir_all(&work_dir).map_err(|e| format!("create work_dir: {e}"))?;
    let mut emu = CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(work_dir))?;
    let (client, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()])?;
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
    let telemetry = emu.telemetry();
    emu.shutdown();
    println!(
        "{}",
        serde_json::json!({
            "self_test": "ok",
            "probed_region": probed[0].region,
            "state": progress.state,
            "predictions_count": pred.row_count,
            "edge_invocations": telemetry.edge_invocations,
            "engine_invocations": telemetry.engine_invocations,
            "eval_invocations": telemetry.eval_invocations,
        })
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
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
        Command::Plan {
            tier,
            effort,
            total_rows,
            fit_rows,
            features,
            ms_per_candidate,
        } => {
            let product_tier = match parse_product_tier(&tier) {
                Ok(t) => t,
                Err(msg) => {
                    eprintln!("error: {msg}");
                    return ExitCode::from(2);
                }
            };
            let effort_tier: CarbonEffortTier = match effort.parse() {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("error: {err} (hint: {})", err.guidance().repair_hint);
                    return ExitCode::from(2);
                }
            };
            let input = CarbonPlanInput {
                product_tier,
                effort: effort_tier,
                total_rows,
                fit_rows,
                features,
                est_ms_per_candidate_1vcpu: ms_per_candidate,
            };
            match plan_carbon_execution(&input) {
                Ok(decision) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&decision).expect("serialize plan decision")
                    );
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
            println!(
                "{}",
                serde_json::to_string_pretty(&verdict).expect("serialize spot verdict")
            );
            ExitCode::SUCCESS
        }
        Command::Serve {
            listen,
            data_dir,
            region,
        } => {
            let mut cfg = CarbonEnterpriseServeConfig::enterprise_default(listen, data_dir);
            cfg.region = region;
            cfg.seal_secret =
                decode_32byte_hex_or_default("CARBON_SEAL_SECRET_HEX", cfg.seal_secret);
            cfg.presign_secret =
                decode_32byte_hex_or_default("CARBON_PRESIGN_SECRET_HEX", cfg.presign_secret);
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
            let Ok(runtime_api) = env::var("LAMBDA_RUNTIME_API") else {
                eprintln!("error: LAMBDA_RUNTIME_API is required for `carbon engine`");
                return ExitCode::from(2);
            };
            let Ok(cas_url) = env::var("CARBON_CAS_URL") else {
                eprintln!("error: CARBON_CAS_URL is required for `carbon engine`");
                return ExitCode::from(2);
            };
            let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
            let eval_invoker = env::var("CARBON_EVAL_INVOKE_URL")
                .ok()
                .map(|u| Arc::new(CarbonHttpEvalInvoker::new(u)) as Arc<_>);
            let seal_secret = decode_32byte_hex_or_default("CARBON_SEAL_SECRET_HEX", [0x77u8; 32]);
            let cas = Arc::new(CarbonHttpCas::new(cas_url, internal_token));
            let worker = match CarbonEngineWorker::new(
                cas,
                eval_invoker,
                CarbonSealKey::from_bytes(seal_secret),
                work_dir,
            ) {
                Ok(w) => w,
                Err(err) => {
                    eprintln!("error: init CarbonEngineWorker: {err}");
                    return ExitCode::from(1);
                }
            };
            match run_lambda_custom_runtime_loop(&runtime_api, None, |inv| {
                worker.handle_invocation_bytes(&inv.payload, inv.deadline_epoch_ms)
            }) {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: carbon engine runtime loop failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Eval { scratch_dir } => {
            let Ok(runtime_api) = env::var("LAMBDA_RUNTIME_API") else {
                eprintln!("error: LAMBDA_RUNTIME_API is required for `carbon eval`");
                return ExitCode::from(2);
            };
            let Ok(cas_url) = env::var("CARBON_CAS_URL") else {
                eprintln!("error: CARBON_CAS_URL is required for `carbon eval`");
                return ExitCode::from(2);
            };
            let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
            let cas = Arc::new(CarbonHttpCas::new(cas_url, internal_token));
            let worker = match CarbonEvalWorker::new(cas, scratch_dir) {
                Ok(w) => w,
                Err(err) => {
                    eprintln!("error: init CarbonEvalWorker: {err}");
                    return ExitCode::from(1);
                }
            };
            match run_lambda_custom_runtime_loop(&runtime_api, None, |inv| {
                worker.handle_wire_bytes(&inv.payload)
            }) {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: carbon eval runtime loop failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
    }
}
