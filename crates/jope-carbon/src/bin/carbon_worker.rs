//! Dedicated minimal AWS Lambda compute bootstrap (`carbon-worker`, `--no-default-features --features eval`).
//!
//! Dispatches `CARBON_ROLE=engine` (sealed orchestrator with runtime Secrets Manager KEK/seal custody)
//! or `CARBON_ROLE=eval` (stateless zero-secret candidate shard evaluator) without linking `clap`,
//! `axum`, `tokio` HTTP servers, `deploy`, or `spec_report` (`GAP-BIN-1`).

use std::env;
use std::process::ExitCode;

use jope_carbon::{
    carbon_https_agent,
    cells::worker_boot::{
        resolve_worker_region, run_engine_lambda_runtime_from_env,
        run_eval_lambda_runtime_from_env, CarbonWorkerBootError,
    },
    observability::write_stdout_line,
    CarbonError, CarbonRole, CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET, CARBON_GIT_SHA,
    CARBON_PKG_VERSION, CARBON_RELEASE_TAG,
};

fn resolve_region() -> String {
    resolve_worker_region()
}

fn resolve_role(args: &[String]) -> Result<CarbonRole, CarbonError> {
    if let Some(first) = args.first() {
        if first == "engine" {
            return Ok(CarbonRole::EngineOrchestrate);
        }
        if first == "eval" {
            return Ok(CarbonRole::EvalBatch);
        }
    }
    match env::var("CARBON_ROLE") {
        Ok(val) if !val.trim().is_empty() => val.parse::<CarbonRole>(),
        _ => Ok(CarbonRole::EngineOrchestrate),
    }
}

fn emit_worker_banner(role_str: &str) -> ExitCode {
    let payload = serde_json::json!({
        "binary": "carbon-worker",
        "version": CARBON_PKG_VERSION,
        "release_tag": CARBON_RELEASE_TAG,
        "git_sha": CARBON_GIT_SHA,
        "build_target": CARBON_BUILD_TARGET,
        "build_profile": CARBON_BUILD_PROFILE,
        "role": role_str,
        "engine_linked": true,
    });
    let _ = write_stdout_line(&payload.to_string());
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        let role_str = resolve_role(&args).map(|r| r.as_str()).unwrap_or("engine");
        return emit_worker_banner(role_str);
    }

    if args.iter().any(|a| a == "--healthcheck") {
        let _agent = carbon_https_agent();
        let role_str = resolve_role(&args).map(|r| r.as_str()).unwrap_or("engine");
        let region = resolve_region();
        let payload = serde_json::json!({
            "status": "ok",
            "binary": "carbon-worker",
            "role": role_str,
            "region": region,
            "git_sha": CARBON_GIT_SHA,
            "tls_provider": "aws_lc_rs",
        });
        let _ = write_stdout_line(&payload.to_string());
        return ExitCode::SUCCESS;
    }

    let role = match resolve_role(&args) {
        Ok(r) => r,
        Err(err) => {
            let guidance = err.guidance();
            eprintln!("error: {err} (hint: {})", guidance.repair_hint);
            return ExitCode::from(2);
        }
    };

    let aws_rt = env::var("AWS_LAMBDA_RUNTIME_API");
    let is_aws_prod = aws_rt.is_ok();
    let runtime_api_env = aws_rt.or_else(|_| env::var("LAMBDA_RUNTIME_API"));
    if args.iter().any(|a| a == "--lambda-runtime") || runtime_api_env.is_ok() {
        let Ok(runtime_api) = runtime_api_env else {
            eprintln!(
                "error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API environment variable is required"
            );
            return ExitCode::from(2);
        };
        let res = match role {
            CarbonRole::EngineOrchestrate => {
                run_engine_lambda_runtime_from_env(&runtime_api, None, is_aws_prod).map(|_| ())
            }
            CarbonRole::EvalBatch => {
                run_eval_lambda_runtime_from_env(&runtime_api, None).map(|_| ())
            }
            other => Err(CarbonWorkerBootError::Config(format!(
                "carbon-worker only executes 'engine' or 'eval' roles (got '{}')",
                other.as_str()
            ))),
        };
        return match res {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: carbon-worker ({}) failed: {err}", role.as_str());
                err.exit_code()
            }
        };
    }

    emit_worker_banner(role.as_str())
}
