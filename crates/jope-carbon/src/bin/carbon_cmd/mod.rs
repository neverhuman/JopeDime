//! Modularized `carbon` CLI command handlers and shared resolution helpers.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{IsTerminal as _, Write as _};
#[cfg(any(feature = "serve", feature = "emu"))]
use std::net::SocketAddr;
pub(crate) use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use base64::Engine as _;
use clap::{ArgAction, CommandFactory as _, Parser, Subcommand};
use jope_api::{
    carbon::{CarbonExplainResponse, CarbonExportLangWire, CarbonTierWire},
    Effort,
};
#[cfg(any(feature = "engine", feature = "eval"))]
use jope_carbon::cells::worker_boot;
#[cfg(feature = "emu")]
use jope_carbon::modes::emu::CarbonRealWorkerBinaries;
use jope_carbon::observability::{emit_cli_error, write_stdout_line, EXIT_AUTH_EXPIRED};
#[cfg(any(
    feature = "engine",
    feature = "eval",
    feature = "serve",
    feature = "emu"
))]
use jope_carbon::CarbonRole;
use jope_carbon::{
    build_client_for_handle, decode_p256_pubkey_b64, estimate_ms_per_candidate_1vcpu,
    evaluate_spot_hybrid_gates, execute_carbon_cli_benchmark_with_token,
    execute_carbon_cli_run_with_ux, execute_carbon_redteam_suite, execute_predict_csv_with_handle,
    execute_scientific_benchmark_suite, execute_self_upgrade_with_options,
    generate_nonlinear_benchmark_rows, inspect_tabular_file_with_tier, list_saved_model_handles,
    list_stored_cloud_tokens, load_delimited_table, load_model_handle_by_spec,
    load_stored_cloud_token, now_epoch_ms, parse_flexible_numeric_f64,
    persist_model_handle_with_options, plan_carbon_execution, query_handle_status,
    remove_stored_cloud_token, render_benchmark_markdown, render_pretty_run_scorecard,
    render_redteam_markdown, render_scientific_benchmark_markdown, resolve_objective_argument,
    resolve_public_cloud_endpoint, save_stored_cloud_token, sha256_lower_hex,
    shred_handle_and_local_keys, verified_absolute_jopedime_config_dir,
    verify_ecdsa_p256_spki_pem_signature, CarbonAuthVerifier, CarbonCapabilityClaims,
    CarbonCliRunOptions, CarbonCliRunUxConfig, CarbonEffortTier, CarbonPlanInput,
    CarbonProbedEndpoint, CarbonProductTier, CarbonPublicCloudsManifest,
    CarbonScientificBenchmarkOptions, CarbonTabularInspectReport, CarbonThinClient,
    MonthlyWindowMetrics, SpotAdvisorThresholds, CARBON_BUILD_PROFILE, CARBON_BUILD_TARGET,
    CARBON_GIT_SHA, CARBON_PKG_VERSION, CARBON_RELEASE_TAG, DEFAULT_PUBLIC_CLOUDS_URL,
    FALLBACK_PUBLIC_CLOUDS_URL, MAX_MINT_TOKEN_TTL_SECS,
};
#[cfg(feature = "engine")]
use jope_carbon::{
    execute_carbon_drift_check, execute_carbon_package, execute_carbon_rollback_multi,
    generate_public_clouds_manifest, mint_capability_token_from_vault, resolve_cell_stack_endpoint,
    CarbonDeployOptions, CarbonLiveAliasVersions, CarbonPackageOptions,
};
#[cfg(feature = "emu")]
use jope_carbon::{
    execute_carbon_spec_report, execute_carbon_spec_verify, CarbonEmuConfig,
    CarbonLocalLambdaEmulator, CarbonSpecReportOptions, CarbonSpecVerifyOptions,
};
#[cfg(feature = "serve")]
use jope_carbon::{CarbonEnterpriseServeConfig, CarbonEnterpriseServer};
use jope_prep::ObjectiveSpec;

pub(crate) mod args;
pub(crate) mod auth_doctor;
pub(crate) mod model_ops;
pub(crate) mod ops_admin;

pub(crate) use args::{
    emit_shell_completions, Command, CommonTrainArgs, FitnessCommand, HandleTargetArgs,
};

const EMBEDDED_RELEASE_SIGNING_PUB_PEM: &str =
    include_str!("../../../../../docs/release/signing.pub");
const EXPECTED_SIGNING_PUB_SHA256: &str =
    "de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e";

fn canonical_embedded_release_signing_pub_pem() -> String {
    EMBEDDED_RELEASE_SIGNING_PUB_PEM.replace("\r\n", "\n")
}

pub(crate) fn parse_product_tier(raw: &str) -> Result<CarbonProductTier, String> {
    match raw.trim() {
        "free" => Ok(CarbonProductTier::Free),
        "pro" => Ok(CarbonProductTier::Pro),
        "enterprise" => Ok(CarbonProductTier::no_limit()),
        other => Err(format!(
            "unknown product tier `{other}` (expected free, pro, or enterprise)"
        )),
    }
}

pub(crate) fn parse_tier_wire(raw: &str) -> Result<CarbonTierWire, String> {
    match raw.trim() {
        "free" => Ok(CarbonTierWire::Free),
        "pro" => Ok(CarbonTierWire::Pro),
        "enterprise" => Ok(CarbonTierWire::Enterprise),
        other => Err(format!(
            "unknown tier `{other}` (expected free, pro, or enterprise)"
        )),
    }
}

pub(crate) fn parse_api_effort(raw: &str) -> Result<Effort, String> {
    match raw.trim() {
        "fast" => Ok(Effort::Fast),
        "medium" => Ok(Effort::Medium),
        "high" => Ok(Effort::High),
        "ultra" => Ok(Effort::Ultra),
        other => Err(format!(
            "unknown effort `{other}` (expected fast, medium, high, or ultra)"
        )),
    }
}

#[cfg(feature = "serve")]
pub(crate) fn decode_required_32byte_hex(env_key: &str) -> Result<[u8; 32], String> {
    let val = env::var(env_key).map_err(|_| format!("missing required {env_key}"))?;
    let raw = hex::decode(val.trim()).map_err(|_| format!("{env_key} must be 64-char hex"))?;
    if raw.len() != 32 {
        return Err(format!("{env_key} must decode to 32 bytes"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

#[cfg(feature = "serve")]
pub(crate) fn decode_or_ephemeral_32byte_secret(
    env_key: &str,
    allow_ephemeral: bool,
) -> Result<[u8; 32], String> {
    match env::var(env_key) {
        Ok(val) if !val.trim().is_empty() => decode_required_32byte_hex(env_key),
        _ if allow_ephemeral => {
            let mut out = [0u8; 32];
            aws_lc_rs::rand::fill(&mut out)
                .map_err(|_| format!("CSPRNG failed generating ephemeral {env_key}"))?;
            Ok(out)
        }
        _ => Err(format!(
            "non-loopback `carbon serve` requires {env_key} (64-char hex) or explicit --dev-ephemeral"
        )),
    }
}

pub(crate) fn resolve_stack_endpoint_if_compiled(
    env_name: &str,
    region: &str,
) -> Result<String, String> {
    #[cfg(feature = "engine")]
    {
        resolve_cell_stack_endpoint(env_name, region)
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (env_name, region);
        Err(
            "--env CloudFormation stack resolution requires the operator build (--features engine); public clients use --cloud <prod|dev> or --endpoint <URL>"
                .to_owned(),
        )
    }
}

pub(crate) fn mint_vault_token_if_compiled(
    env_name: &str,
    region: &str,
    tenant: &str,
    tier: CarbonTierWire,
    ttl_secs: u64,
    admin: bool,
) -> Result<String, String> {
    #[cfg(feature = "engine")]
    {
        mint_capability_token_from_vault(env_name, region, tenant, tier, ttl_secs, admin)
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (env_name, region, tenant, tier, ttl_secs, admin);
        Err(
            "--env Secrets Manager token minting requires the operator build (--features engine); pass CARBON_AUTH_SECRET_HEX or --token <ctk1...>"
                .to_owned(),
        )
    }
}

pub(crate) fn compiled_binary_roles() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut roles = vec!["cli"];
    #[cfg(feature = "engine")]
    roles.push(CarbonRole::EngineOrchestrate.as_str());
    #[cfg(feature = "eval")]
    roles.push(CarbonRole::EvalBatch.as_str());
    #[cfg(feature = "serve")]
    roles.push(CarbonRole::EnterpriseServe.as_str());
    #[cfg(feature = "emu")]
    roles.push(CarbonRole::LocalEmulator.as_str());
    roles
}

pub(crate) fn read_cli_token_source(
    inline_token: Option<String>,
    token_file: Option<PathBuf>,
    token_stdin: bool,
) -> Result<Option<String>, String> {
    if let Some(tok) = inline_token {
        let trimmed = tok.trim().to_owned();
        if trimmed.is_empty() {
            return Err("--token must not be empty".to_owned());
        }
        jope_carbon::observability::log_warn(
            "passing --token on the command line exposes credentials to local process listings; prefer --token-stdin, --token-file <PATH>, or CARBON_AUTH_TOKEN",
        );
        return Ok(Some(trimmed));
    }
    if token_stdin {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin()
            .take(64 * 1024)
            .read_to_string(&mut s)
            .map_err(|e| format!("read token from stdin: {e}"))?;
        let trimmed = s.trim().to_owned();
        if trimmed.is_empty() {
            return Err("stdin token is empty".to_owned());
        }
        return Ok(Some(trimmed));
    }
    if let Some(path) = token_file {
        let raw = if path.as_os_str() == "-" {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin()
                .take(64 * 1024)
                .read_to_string(&mut s)
                .map_err(|e| format!("read token from stdin: {e}"))?;
            s
        } else {
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?
        };
        let trimmed = raw.trim().to_owned();
        if trimmed.is_empty() {
            return Err(format!("token file {} is empty", path.display()));
        }
        return Ok(Some(trimmed));
    }
    Ok(None)
}

pub(crate) fn find_first_carbon_in_path() -> Option<PathBuf> {
    let path_os = env::var_os("PATH")?;
    let bin_name = if cfg!(windows) {
        "carbon.exe"
    } else {
        "carbon"
    };
    for dir in env::split_paths(&path_os) {
        let cand = dir.join(bin_name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

#[cfg(feature = "emu")]
pub(crate) struct EphemeralScratchDir {
    pub(crate) path: PathBuf,
    cleanup_on_drop: bool,
}

#[cfg(feature = "emu")]
impl EphemeralScratchDir {
    pub(crate) fn create(prefix: &str) -> Result<Self, String> {
        let parent = verified_absolute_jopedime_config_dir()
            .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        fs::create_dir_all(&parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
        let dir = parent.join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            now_epoch_ms()
        ));
        fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));
        }
        Ok(Self {
            path: dir,
            cleanup_on_drop: true,
        })
    }

    pub(crate) fn existing_or_create(
        explicit: Option<PathBuf>,
        prefix: &str,
    ) -> Result<Self, String> {
        match explicit {
            Some(p) => {
                fs::create_dir_all(&p).map_err(|e| format!("create {}: {e}", p.display()))?;
                Ok(Self {
                    path: p,
                    cleanup_on_drop: false,
                })
            }
            None => Self::create(prefix),
        }
    }
}

#[cfg(feature = "emu")]
impl Drop for EphemeralScratchDir {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(feature = "emu")]
pub(crate) fn resolve_real_worker_binaries(
    edge_override: Option<PathBuf>,
    worker_override: Option<PathBuf>,
) -> Result<CarbonRealWorkerBinaries, String> {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let edge_name = if cfg!(windows) {
        "carbon-edge.exe"
    } else {
        "carbon-edge"
    };
    let worker_name = if cfg!(windows) {
        "carbon-worker.exe"
    } else {
        "carbon-worker"
    };
    let edge_bin = edge_override
        .or_else(|| env::var_os("CARBON_EDGE_BIN").map(PathBuf::from))
        .or_else(|| {
            exe_dir
                .as_ref()
                .map(|d| d.join(edge_name))
                .filter(|p| p.is_file())
        })
        .ok_or_else(|| {
            format!(
                "could not locate `{edge_name}` binary (pass --edge-bin <PATH> or set CARBON_EDGE_BIN)"
            )
        })?;
    let worker_bin = worker_override
        .or_else(|| env::var_os("CARBON_WORKER_BIN").map(PathBuf::from))
        .or_else(|| {
            exe_dir
                .as_ref()
                .map(|d| d.join(worker_name))
                .filter(|p| p.is_file())
        })
        .ok_or_else(|| {
            format!(
                "could not locate `{worker_name}` binary (pass --worker-bin <PATH> or set CARBON_WORKER_BIN)"
            )
        })?;
    Ok(CarbonRealWorkerBinaries::new(edge_bin, worker_bin))
}

pub(crate) struct LocalEmuGuard {
    #[cfg(feature = "emu")]
    emu: Option<CarbonLocalLambdaEmulator>,
    #[cfg(feature = "emu")]
    _scratch: Option<EphemeralScratchDir>,
}

impl Drop for LocalEmuGuard {
    fn drop(&mut self) {
        #[cfg(feature = "emu")]
        if let Some(mut emu) = self.emu.take() {
            emu.shutdown();
        }
    }
}

pub(crate) fn start_local_emu_session(
    tenant: &str,
) -> Result<(LocalEmuGuard, String, String), String> {
    #[cfg(feature = "emu")]
    {
        let scratch = EphemeralScratchDir::create(".local-emu")?;
        let mut cfg = CarbonEmuConfig::for_dir(scratch.path.clone());
        let want_real = env::var("CARBON_LOCAL_REAL_WORKERS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if want_real {
            cfg = cfg.with_real_worker_binaries(resolve_real_worker_binaries(None, None)?);
        }
        let emu = CarbonLocalLambdaEmulator::start(cfg)?;
        let ep = emu.base_url();
        let tok = emu.mint_capability_token(tenant, CarbonTierWire::Pro, false)?;
        Ok((
            LocalEmuGuard {
                emu: Some(emu),
                _scratch: Some(scratch),
            },
            ep,
            tok,
        ))
    }
    #[cfg(not(feature = "emu"))]
    {
        let _ = tenant;
        Err(
            "--local offline emulator requires a build with `--features emu`; use `carbon inspect` / `--dry-run` offline or connect to `--cloud prod`"
                .to_owned(),
        )
    }
}

pub(crate) fn resolve_explicit_tier_flag(
    free: bool,
    pro: bool,
    tier: Option<&str>,
) -> Result<Option<CarbonTierWire>, String> {
    if free {
        Ok(Some(CarbonTierWire::Free))
    } else if pro {
        Ok(Some(CarbonTierWire::Pro))
    } else if let Some(t) = tier {
        parse_tier_wire(t).map(Some)
    } else {
        Ok(None)
    }
}

pub(crate) struct ResolvedCliTarget {
    pub(crate) endpoints: Vec<String>,
    pub(crate) pinned_keks: BTreeMap<String, String>,
    pub(crate) probed_client: CarbonThinClient,
    pub(crate) first_probe: CarbonProbedEndpoint,
    pub(crate) tier: CarbonTierWire,
    pub(crate) tenant_id: String,
    pub(crate) auth_token: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_cli_target_and_auth(
    mut endpoints: Vec<String>,
    mut cloud: Option<String>,
    clouds_manifest: Option<&str>,
    environment: Option<&str>,
    region: &str,
    explicit_tier: Option<CarbonTierWire>,
    explicit_tenant: Option<String>,
    default_tenant: &str,
    inline_token: Option<String>,
    token_file: Option<PathBuf>,
    token_stdin: bool,
    default_tier_when_self_hosted: CarbonTierWire,
) -> Result<ResolvedCliTarget, (ExitCode, String)> {
    if endpoints.is_empty() && cloud.is_none() && environment.is_none() {
        if let Ok(env_ep) = env::var("CARBON_ENDPOINT") {
            let trimmed = env_ep.trim();
            if !trimmed.is_empty() {
                endpoints.push(trimmed.to_owned());
            }
        }
        if endpoints.is_empty() {
            cloud = Some("prod".to_owned());
        }
    }

    let mut pinned_keks: BTreeMap<String, String> = BTreeMap::new();
    if let Some(cloud_name) = &cloud {
        if endpoints.is_empty() || clouds_manifest.is_some() {
            let entry = resolve_public_cloud_endpoint(cloud_name, clouds_manifest)
                .map_err(|e| (ExitCode::from(2), e))?;
            pinned_keks = entry.pinned_keks_by_endpoint();
            if endpoints.is_empty() {
                endpoints = entry.candidate_endpoints();
            }
        }
    }
    if let Some(env_name) = environment {
        if endpoints.is_empty() {
            let url = resolve_stack_endpoint_if_compiled(env_name, region)
                .map_err(|e| (ExitCode::from(2), e))?;
            endpoints.push(url);
        }
    }
    if endpoints.is_empty() {
        return Err((
            ExitCode::from(2),
            "pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>".to_owned(),
        ));
    }

    let (probed_client, probed) =
        CarbonThinClient::connect_fastest_pinned(endpoints.clone(), &pinned_keks)
            .map_err(|e| (ExitCode::from(1), e))?;
    let first_probe = probed
        .first()
        .cloned()
        .ok_or_else(|| (ExitCode::from(1), "empty probe response".to_owned()))?;
    endpoints = probed.iter().map(|p| p.base_url.clone()).collect();

    let now_secs = now_epoch_ms() / 1_000;
    let cli_tok = read_cli_token_source(inline_token, token_file, token_stdin)
        .map_err(|e| (ExitCode::from(2), e))?;
    let env_tok = env::var("CARBON_AUTH_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    let has_local_hmac_secret = env::var("CARBON_AUTH_SECRET_HEX")
        .ok()
        .is_some_and(|s| !s.trim().is_empty());

    let mut active_token: Option<String> = if explicit_tier == Some(CarbonTierWire::Free) {
        cli_tok
    } else {
        cli_tok.or(env_tok)
    };
    if active_token.is_none() {
        if let Some(lookup_cloud) = cloud.as_deref() {
            if let Ok(Some(stored)) = load_stored_cloud_token(lookup_cloud) {
                if explicit_tier == Some(CarbonTierWire::Free) {
                    if stored.tier == CarbonTierWire::Free && stored.exp_epoch_secs > now_secs {
                        active_token = Some(stored.capability_token);
                    }
                } else if stored.exp_epoch_secs > now_secs {
                    active_token = Some(stored.capability_token);
                } else if matches!(
                    explicit_tier,
                    Some(CarbonTierWire::Pro | CarbonTierWire::Enterprise)
                ) {
                    return Err((
                        ExitCode::from(2),
                        format!(
                            "stored capability token for cloud '{lookup_cloud}' expired at epoch {}; run 'carbon login --cloud {lookup_cloud} --token <ctk1...>' or pass --token",
                            stored.exp_epoch_secs
                        ),
                    ));
                }
            }
        }
    }

    let mut effective_tenant = explicit_tenant
        .clone()
        .unwrap_or_else(|| default_tenant.to_owned());
    let mut effective_tier = explicit_tier.unwrap_or({
        if first_probe.enterprise_self_hosted || has_local_hmac_secret || environment.is_some() {
            default_tier_when_self_hosted
        } else {
            CarbonTierWire::Free
        }
    });

    if let Some(tok) = &active_token {
        let claims = CarbonCapabilityClaims::decode_unverified(tok)
            .map_err(|e| (ExitCode::from(2), e.to_string()))?;
        if now_secs > claims.exp_epoch_secs {
            return Err((
                ExitCode::from(2),
                format!(
                    "capability token for tenant '{}' has expired (exp={})",
                    claims.tenant_id, claims.exp_epoch_secs
                ),
            ));
        }
        if let Some(exp_tenant) = &explicit_tenant {
            if exp_tenant.trim() != claims.tenant_id {
                return Err((
                    ExitCode::from(2),
                    format!(
                        "--tenant '{}' does not match token tenant_id '{}' (omit --tenant to auto-use the token's tenant_id)",
                        exp_tenant.trim(),
                        claims.tenant_id
                    ),
                ));
            }
        }
        effective_tenant = claims.tenant_id;
        if let Some(req_tier) = explicit_tier {
            if req_tier == CarbonTierWire::Pro && claims.tier == CarbonTierWire::Free {
                return Err((
                    ExitCode::from(2),
                    "flag --pro requires a Pro or Enterprise capability token, but provided token has tier 'free'".to_owned(),
                ));
            }
            effective_tier = req_tier;
        } else {
            effective_tier = claims.tier;
        }
    } else if let Some(env_name) = environment {
        if effective_tier != CarbonTierWire::Free && !has_local_hmac_secret {
            let minted = mint_vault_token_if_compiled(
                env_name,
                region,
                &effective_tenant,
                effective_tier,
                3_600,
                false,
            )
            .map_err(|e| (ExitCode::from(2), e))?;
            active_token = Some(minted);
        }
    } else if !first_probe.enterprise_self_hosted
        && !has_local_hmac_secret
        && matches!(
            effective_tier,
            CarbonTierWire::Pro | CarbonTierWire::Enterprise
        )
    {
        let cloud_hint = cloud.as_deref().unwrap_or("prod");
        return Err((
            ExitCode::from(2),
            format!(
                "Pro/Enterprise tier on public cloud '{cloud_hint}' requires a capability token; pass --token <ctk1...> / --token-file <PATH> or run 'carbon login --cloud {cloud_hint} --token <ctk1...>'"
            ),
        ));
    }

    Ok(ResolvedCliTarget {
        endpoints,
        pinned_keks,
        probed_client,
        first_probe,
        tier: effective_tier,
        tenant_id: effective_tenant,
        auth_token: active_token,
    })
}

#[cfg(feature = "engine")]
pub(crate) fn run_engine_runtime(
    runtime_api: &str,
    work_dir: PathBuf,
    is_aws_prod: bool,
) -> ExitCode {
    match worker_boot::run_engine_lambda_runtime_from_env(runtime_api, Some(work_dir), is_aws_prod)
    {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            err.exit_code()
        }
    }
}

#[cfg(feature = "eval")]
pub(crate) fn run_eval_runtime(runtime_api: &str, scratch_dir: PathBuf) -> ExitCode {
    match worker_boot::run_eval_lambda_runtime_from_env(runtime_api, Some(scratch_dir)) {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            err.exit_code()
        }
    }
}

#[cfg(any(feature = "engine", feature = "eval"))]
pub(crate) fn resolve_lambda_runtime_api() -> (Result<String, env::VarError>, bool) {
    let aws = env::var("AWS_LAMBDA_RUNTIME_API");
    let is_aws_prod = aws.is_ok();
    (aws.or_else(|_| env::var("LAMBDA_RUNTIME_API")), is_aws_prod)
}

pub(crate) fn render_interactive_quickstart_banner() -> String {
    let ver = CARBON_PKG_VERSION;
    let tag = CARBON_RELEASE_TAG;
    [
        format!("Carbon v{ver} ({tag}) — Zero-Knowledge Serverless Symbolic GP"),
        String::new(),
        "Quick Start:".to_owned(),
        "  carbon inspect data.csv                     Instant local dataset & cost preflight (zero network)".to_owned(),
        "  carbon fit data.csv --target y              Train encrypted model & save 0600 local handle".to_owned(),
        "  carbon predict test.csv --out preds.csv     Score batch predictions in zero-knowledge mode".to_owned(),
        "  carbon explain / carbon export --lang py    Inspect feature importances or export standalone code".to_owned(),
        "  carbon quickstart                           Run 1-command E2E live encrypted demo".to_owned(),
        String::new(),
        "Account & Diagnostics:".to_owned(),
        "  carbon register --free | carbon login       Manage 0600 capability tokens".to_owned(),
        "  carbon doctor          | carbon upgrade     Verify KEK pins, RTT, and ECDSA-signed CLI updates".to_owned(),
        String::new(),
        "Run `carbon --help` for all commands or `carbon version` for JSON build metadata.".to_owned(),
    ]
    .join("\n")
}

pub(crate) fn write_optional_outputs(
    json_out: Option<PathBuf>,
    markdown_out: Option<PathBuf>,
    json_text: &str,
    md_text: &str,
) -> Result<(), String> {
    if let Some(p) = json_out {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&p, format!("{json_text}\n"))
            .map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    if let Some(p) = markdown_out {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&p, md_text).map_err(|e| format!("write {}: {e}", p.display()))?;
    }
    Ok(())
}
