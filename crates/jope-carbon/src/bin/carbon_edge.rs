//! Regional public-facing `carbon-edge` gateway bootstrap (`--no-default-features --features edge`).
//!
//! Compiles strictly without `jope-service`, `jope-runtime`, `feat-math`, or neural weights, and
//! fails closed on real AWS Lambda (`AWS_LAMBDA_RUNTIME_API`) if `CARBON_PRESIGN_SECRET_HEX`,
//! `CARBON_AUTH_SECRET_HEX`, or `CARBON_ENGINE_KEK_PUBKEY_B64` are unconfigured (`RT-05`).

use std::env;
use std::process::ExitCode;

use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use jope_carbon::{
    carbon_https_agent, decode_p256_pubkey_b64, open_engine_invoker, open_remote_cas,
    parse_s3_cas_url, run_lambda_custom_runtime_loop, CarbonAuthVerifier, CarbonEdgeConfig,
    CarbonEdgeRouter, CarbonError, CarbonPresigner, CarbonRole,
};

fn decode_required_32byte_hex(env_key: &str) -> Result<[u8; 32], String> {
    let val = env::var(env_key).map_err(|_| format!("missing required {env_key}"))?;
    let raw = hex::decode(val.trim()).map_err(|_| format!("{env_key} must be 64-char hex"))?;
    if raw.len() != 32 {
        return Err(format!("{env_key} must decode to exactly 32 bytes"));
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

    let presign_secret = if is_aws_production {
        decode_required_32byte_hex("CARBON_PRESIGN_SECRET_HEX")?
    } else {
        decode_32byte_hex_or_default("CARBON_PRESIGN_SECRET_HEX", [0x33u8; 32])?
    };

    let auth_verifier = if is_aws_production {
        let auth_hex = env::var("CARBON_AUTH_SECRET_HEX")
            .map_err(|_| "missing required CARBON_AUTH_SECRET_HEX on AWS Lambda".to_owned())?;
        Some(CarbonAuthVerifier::from_hex(&auth_hex).map_err(|e| e.to_string())?)
    } else if let Ok(auth_hex) = env::var("CARBON_AUTH_SECRET_HEX") {
        if !auth_hex.trim().is_empty() {
            Some(CarbonAuthVerifier::from_hex(&auth_hex).map_err(|e| e.to_string())?)
        } else {
            None
        }
    } else {
        None
    };

    let engine_kek_pubkey_b64 = if is_aws_production {
        let b64 = env::var("CARBON_ENGINE_KEK_PUBKEY_B64").map_err(|_| {
            "missing required CARBON_ENGINE_KEK_PUBKEY_B64 on AWS Lambda".to_owned()
        })?;
        let _ = decode_p256_pubkey_b64(&b64).map_err(|e| e.to_string())?;
        Some(b64.trim().to_owned())
    } else if let Ok(b64) = env::var("CARBON_ENGINE_KEK_PUBKEY_B64") {
        if !b64.trim().is_empty() {
            let _ = decode_p256_pubkey_b64(&b64).map_err(|e| e.to_string())?;
            Some(b64.trim().to_owned())
        } else {
            None
        }
    } else {
        None
    };

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
            presigner: CarbonPresigner::from_secret(presign_secret),
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
