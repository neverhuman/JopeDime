//! Carbon zero-knowledge serverless cell runtime, `r882` neural fusion orchestrator, adaptive
//! multi-class execution planner, authenticated state sealer, and Enterprise/CI server ([ADR-0007]).
//!
//! Organized into explicit trust-boundary packages:
//! - [`security`]: `CUEN\x01`/`CUDK\x01` per-user `ECDH P-256 + HKDF-SHA256 + AES-256-GCM` envelopes,
//!   `HMAC-SHA256` capability tokens (`ctk1.*`), `CCKP\x02` domain-separated sealing, runtime
//!   Secrets Manager custody, and size-pinned SigV4.
//! - [`cells`]: Strictly feature-isolated `edge` (`--no-default-features --features edge`), `engine`, and `eval` workers.
//! - [`storage`]: Tenant/upload-scoped `blobs/<tenant>/<upload>/<sha256>` and injective `runs/<tenant>/<run>/<leaf>` CAS backends.
//! - [`modes`]: Adaptive execution planner, 3-Gate Spot-Hybrid advisor, `carbon serve` VPC server, and `carbon emu`.
//! - [`cli`]: Zero-knowledge `CarbonThinClient`, Free/Pro `r882` benchmark runner, and 16-vector red-team suite.
//!
//! [ADR-0007]: ../../docs/adr/ADR-0007-carbon-lambda-cells.md

/// Workspace package version of `jope-carbon`.
pub const CARBON_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Full source Git commit SHA embedded at compile time (`build.rs`).
pub const CARBON_GIT_SHA: &str = env!("CARBON_GIT_SHA");
/// Target triple (`x86_64-unknown-linux-musl`, `aarch64-apple-darwin`, etc.) embedded at compile time.
pub const CARBON_BUILD_TARGET: &str = env!("CARBON_BUILD_TARGET");
/// Cargo profile (`carbon-cli`, `dist`, `release`, `debug`) embedded at compile time.
pub const CARBON_BUILD_PROFILE: &str = env!("CARBON_BUILD_PROFILE");
/// Canonical Carbon CLI distribution release tag (`carbon-cli-v1.1.0`).
pub const CARBON_RELEASE_TAG: &str = env!("CARBON_RELEASE_TAG");

/// Return a deterministic provenance `User-Agent` header for outbound Carbon HTTP requests.
pub fn carbon_user_agent() -> String {
    let sha12: String = CARBON_GIT_SHA.chars().take(12).collect();
    format!(
        "carbon/{} ({}; tag/{}; sha/{})",
        CARBON_PKG_VERSION, CARBON_BUILD_TARGET, CARBON_RELEASE_TAG, sha12
    )
}

pub mod cells;
#[cfg(feature = "cli")]
pub mod cli;
pub mod error;
mod home_path;
pub mod modes;
pub mod role;
pub mod security;
pub mod storage;
pub mod tls;

#[cfg(feature = "engine")]
pub use cells::engine::CarbonEngineWorker;
#[cfg(feature = "engine")]
pub use cells::eval::{
    open_eval_invoker, CarbonAwsLambdaEvalInvoker, CarbonEvalDecryptionGrant, CarbonEvalInvoker,
    CarbonEvalWireRequest, CarbonEvalWorker, CarbonHttpEvalInvoker,
};
#[cfg(feature = "edge")]
pub use cells::{
    edge::{CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayRequest, CarbonGatewayResponse},
    engine_invoker::{
        now_epoch_ms, open_engine_invoker, CarbonAwsLambdaEngineInvoker, CarbonEngineInvoker,
        CarbonEngineWireCommand, CarbonHttpEngineInvoker,
    },
    lambda_runtime::{run_lambda_custom_runtime_loop, CarbonLambdaInvocation},
};
#[cfg(all(feature = "cli", feature = "engine"))]
pub use cli::deploy::{
    build_deterministic_single_bin_targz, build_deterministic_zip_bytes, execute_carbon_deploy,
    execute_carbon_package, execute_carbon_rollback, execute_carbon_rollback_multi,
    expand_region_targets, generate_public_clouds_manifest, mint_capability_token_from_vault,
    resolve_cell_stack_endpoint, resolve_cell_stack_public_endpoint,
    synthesize_cell_template_for_release, verify_arm64_lambda_binary, CarbonDeployOptions,
    CarbonDeployReport, CarbonLiveAliasVersions, CarbonPackageOptions, CarbonRegionDeployReceipt,
    CarbonReleaseManifest, CarbonRollbackOptions, CarbonRollbackReceipt, CarbonZipEntry,
    CANONICAL_17_REGIONS, CLOUDFORMATION_CELL_TEMPLATE_JSON,
};
#[cfg(all(feature = "cli", feature = "emu"))]
pub use cli::spec_report::{
    execute_carbon_spec_report, execute_carbon_spec_verify, verify_pdf_bytes, verify_png_bytes,
    CarbonSpecReportOptions, CarbonSpecVerifyOptions, MIN_SPEC_PDF_PAGES, SPEC_RELEASE_TAG,
};
#[cfg(feature = "cli")]
pub use cli::{
    client::{
        list_stored_cloud_tokens, load_stored_cloud_token, remove_stored_cloud_token,
        resolve_public_cloud_endpoint, save_stored_cloud_token, send_raw_edge_http,
        send_raw_put_url, verified_absolute_jopedime_config_dir, CarbonClientPreparedDataset,
        CarbonCloudCellEntry, CarbonCloudRegionPin, CarbonProbedEndpoint, CarbonProgressObserver,
        CarbonPublicCloudsManifest, CarbonStoredCloudCredential, CarbonThinClient,
        DEFAULT_PUBLIC_CLOUDS_URL, FALLBACK_PUBLIC_CLOUDS_URL,
    },
    redteam::{
        execute_carbon_redteam_suite, render_redteam_markdown, CarbonRedTeamSuiteReport,
        CarbonRedTeamVectorResult,
    },
    runner::{
        execute_carbon_cli_benchmark, execute_carbon_cli_benchmark_with_token,
        execute_carbon_cli_run, execute_carbon_cli_run_with_ux, generate_nonlinear_benchmark_rows,
        load_numeric_csv, render_benchmark_markdown, CarbonBenchScenarioSummary,
        CarbonBenchmarkSuiteReport, CarbonCliRunOptions, CarbonCliRunReceipt, CarbonCliRunUxConfig,
    },
};
pub use error::{CarbonError, CarbonGuidance, CarbonResult, CARBON_DOCS_URL};
#[cfg(feature = "emu")]
pub use modes::emu::{CarbonEmuConfig, CarbonEmuTelemetrySnapshot, CarbonLocalLambdaEmulator};
#[cfg(feature = "serve")]
pub use modes::serve::{
    CarbonEnterpriseServeConfig, CarbonEnterpriseServer, CarbonEnterpriseServerHandle,
};
pub use modes::{
    planner::{
        compute_live_run_cost, estimate_working_set_mb, plan_carbon_execution, CarbonExecutionMode,
        CarbonIngressCostMode, CarbonLiveCostInputs, CarbonPlanDecision, CarbonPlanInput,
        CarbonRunCostBreakdown, EvalMemoryClass, APIGW_HTTP_API_USD_PER_UNIT,
        AWS_INTERNET_EGRESS_USD_PER_GB, DEPLOYED_EVAL_MEMORY_MB, EDGE_MEMORY_MB, ENGINE_MEMORY_MB,
        LAMBDA_ARM64_USD_PER_GB_SEC, LAMBDA_USD_PER_INVOCATION, S3_STANDARD_GET_USD_PER_REQ,
        S3_STANDARD_PUT_LIST_USD_PER_REQ, S3_STANDARD_USD_PER_GB_MONTH,
    },
    spot_advisor::{
        evaluate_spot_hybrid_gates, MonthlyWindowMetrics, SpotAdvisorRecommendation,
        SpotAdvisorThresholds, SpotGateVerdict,
    },
};
pub use role::{CarbonEffortTier, CarbonProductTier, CarbonRole};
pub use security::auth::{
    validate_strict_run_id, validate_strict_tenant_id, MAX_MINT_TOKEN_TTL_SECS,
};
#[cfg(feature = "edge")]
pub use security::{
    auth::{CarbonAuthVerifier, CarbonCapabilityClaims, CarbonRateLimiter},
    seal::{CarbonSealDomain, CarbonSealKey, SealedCheckpointEnvelope},
    secrets_vault::{
        fetch_secretsmanager_secret_string, load_cached_runtime_secrets, parse_secret_bundle_json,
        resolve_edge_secrets, resolve_engine_secrets, CarbonRuntimeSecrets,
    },
    sigv4::{
        compute_backoff_with_jitter_ms, format_amz_timestamp_utc, sign_aws_v4_headers,
        sign_aws_v4_headers_with_extra, url_encode_component, url_encode_s3_key,
        CarbonAwsCredentials, CarbonPresigner,
    },
    user_envelope::{
        decode_p256_pubkey_b64, decrypt_cuen_payload, decrypt_cuen_with_slot_subkey,
        derive_cuen_slot_subkey, encrypt_cuen_payload, generate_random_u_dek,
        unwrap_u_dek_with_kek, verify_ecdsa_p256_spki_pem_signature, wrap_u_dek_to_pubkey,
        CarbonEncryptedPredictPayload, CarbonEngineKek, CAS_BLOB_CUEN_SCOPE,
    },
};
pub use storage::cas::{
    canonical_blob_scope, canonical_run_state_key, sha256_lower_hex, CarbonCasStore, CarbonFsCas,
};
#[cfg(feature = "edge")]
pub use storage::cas::{open_remote_cas, parse_s3_cas_url, CarbonHttpCas, CarbonS3Cas};
#[cfg(feature = "edge")]
pub use tls::{carbon_https_agent, carbon_probe_https_agent, load_custom_pem_root_certs};
