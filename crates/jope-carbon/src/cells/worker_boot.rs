//! Shared `engine` and `eval` AWS Lambda custom-runtime bootstrap (`carbon-worker` & `carbon`).

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    cells::{
        engine::CarbonEngineWorker, eval::open_eval_invoker, eval::CarbonEvalWorker,
        lambda_runtime::run_lambda_custom_runtime_loop,
    },
    security::secrets_vault::resolve_engine_secrets,
    storage::cas::open_remote_cas,
};

/// Typed distinction between configuration errors (`ExitCode(2)`) and runtime loop errors (`ExitCode(1)`).
#[derive(Debug)]
pub enum CarbonWorkerBootError {
    Config(String),
    Runtime(String),
}

impl CarbonWorkerBootError {
    pub fn exit_code(&self) -> std::process::ExitCode {
        match self {
            Self::Config(_) => std::process::ExitCode::from(2),
            Self::Runtime(_) => std::process::ExitCode::from(1),
        }
    }
}

impl std::fmt::Display for CarbonWorkerBootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(msg) | Self::Runtime(msg) => f.write_str(msg),
        }
    }
}

/// Resolve active AWS region from `CARBON_REGION` or `AWS_REGION` (defaulting to `us-east-2`).
pub fn resolve_worker_region() -> String {
    match env::var("CARBON_REGION").or_else(|_| env::var("AWS_REGION")) {
        Ok(val) if !val.trim().is_empty() => val,
        _ => "us-east-2".to_owned(),
    }
}

/// Boot and run the `CARBON_ROLE=engine` AWS Lambda Custom Runtime event loop from environment config.
pub fn run_engine_lambda_runtime_from_env(
    runtime_api: &str,
    work_dir_override: Option<PathBuf>,
    is_aws_production: bool,
) -> Result<usize, CarbonWorkerBootError> {
    let region = resolve_worker_region();
    let cas_url = env::var("CARBON_CAS_URL").map_err(|_| {
        CarbonWorkerBootError::Config("missing CARBON_CAS_URL for carbon engine runtime".to_owned())
    })?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let (seal_key, engine_kek) = resolve_engine_secrets(&region, is_aws_production)
        .map_err(CarbonWorkerBootError::Config)?;
    let cas = open_remote_cas(&cas_url, &region, &internal_token)
        .map_err(CarbonWorkerBootError::Runtime)?;
    let eval_invoker = env::var("CARBON_EVAL_INVOKE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|u| open_eval_invoker(&u, &region));
    let work_root = work_dir_override.unwrap_or_else(|| {
        env::var("CARBON_WORK_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| env::temp_dir().join("carbon-engine-work"))
    });
    let force_hybrid = env::var("CARBON_FORCE_HYBRID_FANOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let worker = CarbonEngineWorker::new(cas, eval_invoker, seal_key, work_root)
        .map_err(CarbonWorkerBootError::Runtime)?
        .with_engine_kek(engine_kek.map(Arc::new))
        .with_forced_hybrid_fanout(force_hybrid);

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_invocation_bytes(&inv.payload, inv.deadline_epoch_ms)
    })
    .map_err(CarbonWorkerBootError::Runtime)
}

/// Boot and run the `CARBON_ROLE=eval` AWS Lambda Custom Runtime event loop from environment config.
pub fn run_eval_lambda_runtime_from_env(
    runtime_api: &str,
    scratch_dir_override: Option<PathBuf>,
) -> Result<usize, CarbonWorkerBootError> {
    let region = resolve_worker_region();
    let cas_url = env::var("CARBON_CAS_URL").map_err(|_| {
        CarbonWorkerBootError::Config("missing CARBON_CAS_URL for carbon eval runtime".to_owned())
    })?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let cas = open_remote_cas(&cas_url, &region, &internal_token)
        .map_err(CarbonWorkerBootError::Runtime)?;
    let scratch_root = scratch_dir_override.unwrap_or_else(|| {
        env::var("CARBON_SCRATCH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| env::temp_dir().join("carbon-eval-scratch"))
    });
    let worker =
        CarbonEvalWorker::new(cas, scratch_root).map_err(CarbonWorkerBootError::Runtime)?;

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        worker.handle_wire_bytes(&inv.payload)
    })
    .map_err(CarbonWorkerBootError::Runtime)
}
