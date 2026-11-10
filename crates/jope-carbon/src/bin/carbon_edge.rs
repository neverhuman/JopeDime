//! Regional public-facing `carbon-edge` gateway bootstrap (`--no-default-features --features edge`).
//!
//! Compiles strictly without `jope-service`, `jope-runtime`, `feat-math`, or neural weights.

use std::env;
use std::process::ExitCode;

use jope_carbon::{carbon_https_agent, CarbonError, CarbonRole};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--version" || a == "-V") {
        println!(
            "{}",
            serde_json::json!({
                "binary": "carbon-edge",
                "version": env!("CARGO_PKG_VERSION"),
                "role": CarbonRole::Edge.as_str(),
                "engine_linked": false,
            })
        );
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|a| a == "--healthcheck") {
        // Initialize the aws_lc_rs TLS provider to prove single-provider TLS readiness.
        let _agent = carbon_https_agent();
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "role": CarbonRole::Edge.as_str(),
                "tls_provider": "aws_lc_rs",
            })
        );
        return ExitCode::SUCCESS;
    }

    let err = CarbonError::InvalidRoleOrCommand {
        value: args.join(" "),
    };
    let guidance = err.guidance();
    eprintln!("error: {err} (hint: {})", guidance.repair_hint);
    ExitCode::from(2)
}
