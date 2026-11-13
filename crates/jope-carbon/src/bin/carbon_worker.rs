//! Dedicated minimal AWS Lambda compute bootstrap (`carbon-worker`, `--no-default-features --features eval`).
//!
//! Dispatches `CARBON_ROLE=engine` (sealed orchestrator with runtime Secrets Manager KEK/seal custody)
//! or `CARBON_ROLE=eval` (stateless zero-secret candidate shard evaluator) without linking `clap`,
//! `axum`, `tokio` HTTP servers, `deploy`, or `spec_report` (`GAP-BIN-1`).

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use jope_carbon::{
    carbon_https_agent, open_eval_invoker, open_remote_cas, resolve_engine_secrets,
    run_lambda_custom_runtime_loop, CarbonEngineWorker, CarbonError, CarbonEvalWorker, CarbonRole,
    CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET, CARBON_GIT_SHA, CARBON_PKG_VERSION,
    CARBON_RELEASE_TAG,
};

fn resolve_region() -> String {
    match env::var("CARBON_REGION").or_else(|_| env::var("AWS_REGION")) {
        Ok(val) if !val.trim().is_empty() => val,
        _ => "us-east-2".to_owned(),
    }
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

fn run_engine_lambda_loop(runtime_api: &str, is_aws_production: bool) -> Result<(), String> {
    let region = resolve_region();
    let cas_url = env::var("CARBON_CAS_URL")
        .map_err(|_| "missing CARBON_CAS_URL for carbon-worker engine runtime".to_owned())?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let (seal_key, engine_kek) = resolve_engine_secrets(&region, is_aws_production)?;
    let cas = open_remote_cas(&cas_url, &region, &internal_token)?;
    let eval_invoker = env::var("CARBON_EVAL_INVOKE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|u| open_eval_invoker(&u, &region));
    let work_root = env::var("CARBON_WORK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("carbon-engine-work"));
    let force_hybrid = env::var("CARBON_FORCE_HYBRID_FANOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let worker = CarbonEngineWorker::new(cas, eval_invoker, seal_key, work_root)?
        .with_engine_kek(engine_kek.map(Arc::new))
        .with_forced_hybrid_fanout(force_hybrid);

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_invocation_bytes(&inv.payload, inv.deadline_epoch_ms)
    })
    .map(|_| ())
}

fn run_eval_lambda_loop(runtime_api: &str) -> Result<(), String> {
    let region = resolve_region();
    let cas_url = env::var("CARBON_CAS_URL")
        .map_err(|_| "missing CARBON_CAS_URL for carbon-worker eval runtime".to_owned())?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let cas = open_remote_cas(&cas_url, &region, &internal_token)?;
    let scratch_root = env::var("CARBON_SCRATCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("carbon-eval-scratch"));
    let worker = CarbonEvalWorker::new(cas, scratch_root)?;

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_wire_bytes(&inv.payload)
    })
    .map(|_| ())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        let role_str = resolve_role(&args).map(|r| r.as_str()).unwrap_or("engine");
        println!(
            "{}",
            serde_json::json!({
                "binary": "carbon-worker",
                "version": CARBON_PKG_VERSION,
                "release_tag": CARBON_RELEASE_TAG,
                "git_sha": CARBON_GIT_SHA,
                "build_target": CARBON_BUILD_TARGET,
                "build_profile": CARBON_BUILD_PROFILE,
                "role": role_str,
                "engine_linked": true,
            })
        );
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|a| a == "--healthcheck") {
        let _agent = carbon_https_agent();
        let role_str = resolve_role(&args).map(|r| r.as_str()).unwrap_or("engine");
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "binary": "carbon-worker",
                "role": role_str,
                "git_sha": CARBON_GIT_SHA,
                "tls_provider": "aws_lc_rs",
            })
        );
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
            CarbonRole::EngineOrchestrate => run_engine_lambda_loop(&runtime_api, is_aws_prod),
            CarbonRole::EvalBatch => run_eval_lambda_loop(&runtime_api),
            other => Err(format!(
                "carbon-worker only executes 'engine' or 'eval' roles (got '{}')",
                other.as_str()
            )),
        };
        match res {
            Ok(()) => return ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: carbon-worker ({}) failed: {err}", role.as_str());
                return ExitCode::from(1);
            }
        }
    }

    println!(
        "{}",
        serde_json::json!({
            "binary": "carbon-worker",
            "version": CARBON_PKG_VERSION,
            "release_tag": CARBON_RELEASE_TAG,
            "git_sha": CARBON_GIT_SHA,
            "build_target": CARBON_BUILD_TARGET,
            "build_profile": CARBON_BUILD_PROFILE,
            "role": role.as_str(),
            "engine_linked": true,
        })
    );
    ExitCode::SUCCESS
}
