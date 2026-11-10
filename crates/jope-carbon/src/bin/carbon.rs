//! Multi-role `carbon` CLI and Lambda custom-runtime entrypoint (`engine`, `eval`, `serve`, `emu`,
//! `plan`, and `spot-check`).

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use jope_carbon::{
    evaluate_spot_hybrid_gates, plan_carbon_execution, CarbonEffortTier, CarbonPlanInput,
    CarbonProductTier, CarbonRole, MonthlyWindowMetrics, SpotAdvisorThresholds,
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
    }
}
