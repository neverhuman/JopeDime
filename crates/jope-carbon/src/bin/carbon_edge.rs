//! Regional public-facing `carbon-edge` gateway bootstrap (`--no-default-features --features edge`).
//!
//! Compiles strictly without `jope-service`, `jope-runtime`, `feat-math`, or neural weights, and
//! fails closed on real AWS Lambda (`AWS_LAMBDA_RUNTIME_API`) if runtime secrets (`CARBON_SECRETS_ARN`
//! or explicit dev env vars) are unconfigured (`RT-05`).

use std::env;
use std::process::ExitCode;

use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use jope_carbon::{
    carbon_https_agent, open_engine_invoker, open_remote_cas, parse_s3_cas_url,
    resolve_edge_secrets, run_lambda_custom_runtime_loop, CarbonEdgeConfig, CarbonEdgeRouter,
    CarbonError, CarbonRole, CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET, CARBON_GIT_SHA,
    CARBON_PKG_VERSION, CARBON_RELEASE_TAG,
};

fn run_edge_lambda_loop(runtime_api: &str, is_aws_production: bool) -> Result<(), String> {
    let region = match env::var("CARBON_REGION").or_else(|_| env::var("AWS_REGION")) {
        Ok(val) if !val.trim().is_empty() => val,
        _ => "us-east-2".to_owned(),
    };
    let public_base_url = env::var("CARBON_PUBLIC_BASE_URL").unwrap_or_default();
    let cas_url = env::var("CARBON_CAS_URL")
        .map_err(|_| "missing CARBON_CAS_URL for carbon-edge runtime".to_owned())?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let engine_url = env::var("CARBON_ENGINE_INVOKE_URL")
        .map_err(|_| "missing CARBON_ENGINE_INVOKE_URL for carbon-edge runtime".to_owned())?;

    let (presigner, auth_verifier, engine_kek_pubkey_b64) =
        resolve_edge_secrets(&region, is_aws_production)?;

    let catalog = if let Ok(domain) = env::var("CARBON_GLOBAL_DOMAIN_SUFFIX") {
        CarbonRegionCatalog::default_17_region_catalog(&domain)
    } else {
        CarbonRegionCatalog {
            catalog_version: 1,
            regions: vec![CarbonRegionDescriptor {
                region: region.clone(),
                geography: "americas".to_owned(),
                endpoint_url: public_base_url.clone(),
                priority_rank: 1,
            }],
        }
    };

    let s3_target = parse_s3_cas_url(&cas_url);
    let cas = open_remote_cas(&cas_url, &region, &internal_token)?;
    let engine = open_engine_invoker(&engine_url, &region);

    let router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region,
            public_base_url,
            enterprise_self_hosted: false,
            catalog,
            presigner,
            auth_verifier,
            engine_kek_pubkey_b64,
        },
        cas,
        engine,
    )
    .with_s3_upload_target(s3_target);

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        router.handle_lambda_invocation(&inv.payload)
    })
    .map(|_| ())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!(
            "{}",
            serde_json::json!({
                "binary": "carbon-edge",
                "version": CARBON_PKG_VERSION,
                "release_tag": CARBON_RELEASE_TAG,
                "git_sha": CARBON_GIT_SHA,
                "build_target": CARBON_BUILD_TARGET,
                "build_profile": CARBON_BUILD_PROFILE,
                "role": CarbonRole::Edge.as_str(),
                "engine_linked": false,
            })
        );
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|a| a == "--healthcheck") {
        let _agent = carbon_https_agent();
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "role": CarbonRole::Edge.as_str(),
                "git_sha": CARBON_GIT_SHA,
                "tls_provider": "aws_lc_rs",
            })
        );
        return ExitCode::SUCCESS;
    }

    let aws_rt = env::var("AWS_LAMBDA_RUNTIME_API");
    let is_aws_prod = aws_rt.is_ok();
    let runtime_api_env = aws_rt.or_else(|_| env::var("LAMBDA_RUNTIME_API"));
    if args.iter().any(|a| a == "--lambda-runtime") || runtime_api_env.is_ok() {
        let Ok(runtime_api) = runtime_api_env else {
            eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API environment variable is required");
            return ExitCode::from(2);
        };
        match run_edge_lambda_loop(&runtime_api, is_aws_prod) {
            Ok(()) => return ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: carbon-edge runtime failed: {err}");
                return ExitCode::from(1);
            }
        }
    }

    if args.is_empty() {
        println!(
            "{}",
            serde_json::json!({
                "binary": "carbon-edge",
                "version": CARBON_PKG_VERSION,
                "release_tag": CARBON_RELEASE_TAG,
                "git_sha": CARBON_GIT_SHA,
                "build_target": CARBON_BUILD_TARGET,
                "build_profile": CARBON_BUILD_PROFILE,
                "role": CarbonRole::Edge.as_str(),
                "engine_linked": false,
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
