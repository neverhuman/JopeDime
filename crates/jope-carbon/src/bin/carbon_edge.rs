//! Regional public-facing `carbon-edge` gateway bootstrap (`--no-default-features --features edge`).
//!
//! Compiles strictly without `jope-service`, `jope-runtime`, `feat-math`, or neural weights.
//! Speaks the AWS Lambda Custom Runtime (`2018-06-01`) protocol when `LAMBDA_RUNTIME_API` is set
//! or `--lambda-runtime` is supplied.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use jope_carbon::{
    carbon_https_agent, run_lambda_custom_runtime_loop, CarbonEdgeConfig, CarbonEdgeRouter,
    CarbonError, CarbonGatewayRequest, CarbonHttpCas, CarbonHttpEngineInvoker, CarbonPresigner,
    CarbonRole,
};

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

fn run_edge_lambda_loop(runtime_api: &str) -> Result<(), String> {
    let region = env::var("CARBON_REGION").unwrap_or_else(|_| "us-east-2".to_owned());
    let public_base_url = env::var("CARBON_PUBLIC_BASE_URL").unwrap_or_default();
    let cas_url = env::var("CARBON_CAS_URL")
        .map_err(|_| "missing CARBON_CAS_URL for carbon-edge runtime".to_owned())?;
    let internal_token = env::var("CARBON_INTERNAL_TOKEN").unwrap_or_default();
    let engine_url = env::var("CARBON_ENGINE_INVOKE_URL")
        .map_err(|_| "missing CARBON_ENGINE_INVOKE_URL for carbon-edge runtime".to_owned())?;
    let presign_secret = decode_32byte_hex_or_default("CARBON_PRESIGN_SECRET_HEX", [0x33u8; 32]);

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

    let router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region,
            public_base_url,
            enterprise_self_hosted: false,
            catalog,
            presigner: CarbonPresigner::from_secret(presign_secret),
        },
        Arc::new(CarbonHttpCas::new(cas_url, internal_token)),
        Arc::new(CarbonHttpEngineInvoker::new(engine_url)),
    );

    run_lambda_custom_runtime_loop(runtime_api, None, |inv| {
        let req: CarbonGatewayRequest = serde_json::from_slice(&inv.payload)
            .map_err(|e| format!("decode CarbonGatewayRequest: {e}"))?;
        let resp = router.handle(&req);
        serde_json::to_vec(&resp).map_err(|e| format!("encode CarbonGatewayResponse: {e}"))
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
                "version": env!("CARGO_PKG_VERSION"),
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
                "tls_provider": "aws_lc_rs",
            })
        );
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|a| a == "--lambda-runtime") || env::var("LAMBDA_RUNTIME_API").is_ok() {
        let Ok(runtime_api) = env::var("LAMBDA_RUNTIME_API") else {
            eprintln!("error: LAMBDA_RUNTIME_API environment variable is required");
            return ExitCode::from(2);
        };
        match run_edge_lambda_loop(&runtime_api) {
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
                "version": env!("CARGO_PKG_VERSION"),
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
