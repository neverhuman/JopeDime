//! Read-only CloudFormation/Lambda/SecretsManager drift verification (`--check-drift`),
//! sub-second `:live` alias rollback (`carbon rollback`), stack output inspection, and
//! `carbon-clouds.json` manifest synthesis.

use std::collections::BTreeMap;
use std::thread;

use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::{
        client::{
            CarbonCloudCellEntry, CarbonCloudRegionPin, CarbonPublicCloudsManifest,
            CarbonThinClient,
        },
        deploy::{
            ensure_cell_secrets_in_vault, expand_region_targets, run_aws_allow_missing,
            run_aws_json, CarbonLiveAliasVersions,
        },
    },
    security::sigv4::CarbonAwsCredentials,
};

/// `CarbonRollbackOptions` is a struct configuring `carbon rollback`.
#[derive(Clone, Debug)]
pub struct CarbonRollbackOptions {
    pub environment: String,
    pub region: String,
    pub target_versions: Option<CarbonLiveAliasVersions>,
}

/// `CarbonRollbackReceipt` is a struct recording an atomic `:live` alias rollback.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonRollbackReceipt {
    pub timestamp_epoch_ms: u64,
    pub stack_name: String,
    pub region: String,
    pub restored_versions: CarbonLiveAliasVersions,
    pub ping_verified: bool,
}

/// Per-region read-only drift verification result (`carbon deploy --check-drift`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonDriftRegionResult {
    pub region: String,
    pub stack_name: String,
    pub stack_status: Option<String>,
    pub stack_release_id: Option<String>,
    pub public_edge_endpoint_url: Option<String>,
    pub live_versions: Option<CarbonLiveAliasVersions>,
    pub vault_kek_pubkey_b64: Option<String>,
    pub ping_kek_pubkey_b64: Option<String>,
    pub manifest_kek_pubkey_b64: Option<String>,
    pub server_git_sha: Option<String>,
    pub in_sync: bool,
    pub drift_reasons: Vec<String>,
}

/// Multi-region read-only drift report (`carbon deploy --check-drift`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonDriftReport {
    pub timestamp_epoch_ms: u64,
    pub environment: String,
    pub all_in_sync: bool,
    pub regions: Vec<CarbonDriftRegionResult>,
}

pub(crate) struct StackStateSnapshot {
    pub(crate) status: String,
    pub(crate) outputs: BTreeMap<String, String>,
}

pub(crate) fn read_stack_outputs_map(
    region: &str,
    stack_name: &str,
) -> Result<Option<StackStateSnapshot>, String> {
    let Some(desc) = run_aws_allow_missing(
        region,
        &[
            "cloudformation",
            "describe-stacks",
            "--stack-name",
            stack_name,
        ],
    )?
    else {
        return Ok(None);
    };
    let Some(stack) = desc
        .get("Stacks")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    else {
        return Ok(None);
    };
    let status = stack
        .get("StackStatus")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_owned();
    let mut outputs = BTreeMap::new();
    if let Some(outs) = stack.get("Outputs").and_then(|v| v.as_array()) {
        for item in outs {
            if let (Some(k), Some(v)) = (
                item.get("OutputKey").and_then(|k| k.as_str()),
                item.get("OutputValue").and_then(|v| v.as_str()),
            ) {
                outputs.insert(k.to_owned(), v.to_owned());
            }
        }
    }
    Ok(Some(StackStateSnapshot { status, outputs }))
}

/// Resolve the live `EdgeEndpointUrl` from CloudFormation stack `jopedime-carbon-<env>` in `region`.
pub fn resolve_cell_stack_endpoint(environment: &str, region: &str) -> Result<String, String> {
    let stack_name = format!("jopedime-carbon-{environment}");
    let snap = read_stack_outputs_map(region, &stack_name)?
        .ok_or_else(|| format!("stack '{stack_name}' not found in {region}"))?;
    snap.outputs
        .get("EdgeEndpointUrl")
        .cloned()
        .ok_or_else(|| format!("stack '{stack_name}' in {region} missing EdgeEndpointUrl output"))
}

/// Resolve the live `PublicEdgeEndpointUrl` (falling back to `EdgeEndpointUrl` if absent) from
/// CloudFormation stack `jopedime-carbon-<env>` in `region`.
pub fn resolve_cell_stack_public_endpoint(
    environment: &str,
    region: &str,
) -> Result<String, String> {
    let stack_name = format!("jopedime-carbon-{environment}");
    let snap = read_stack_outputs_map(region, &stack_name)?
        .ok_or_else(|| format!("stack '{stack_name}' not found in {region}"))?;
    snap.outputs
        .get("PublicEdgeEndpointUrl")
        .or_else(|| snap.outputs.get("EdgeEndpointUrl"))
        .cloned()
        .ok_or_else(|| {
            format!(
                "stack '{stack_name}' in {region} missing PublicEdgeEndpointUrl / EdgeEndpointUrl output"
            )
        })
}

pub(crate) fn read_current_live_aliases(
    region: &str,
    stack_name: &str,
) -> Option<CarbonLiveAliasVersions> {
    let read_one = |suffix: &str| -> Option<String> {
        let doc = run_aws_allow_missing(
            region,
            &[
                "lambda",
                "get-alias",
                "--function-name",
                &format!("{stack_name}-{suffix}"),
                "--name",
                "live",
            ],
        )
        .ok()??;
        doc.get("FunctionVersion")?.as_str().map(str::to_owned)
    };
    Some(CarbonLiveAliasVersions {
        edge_version: read_one("edge")?,
        engine_version: read_one("engine")?,
        eval_version: read_one("eval")?,
    })
}

pub(crate) fn decommission_unstacked_prototype_resources(
    region: &str,
    stack_name: &str,
) -> Result<(), String> {
    for suffix in ["edge", "engine", "eval"] {
        let fn_name = format!("{stack_name}-{suffix}");
        let _ = run_aws_allow_missing(
            region,
            &[
                "lambda",
                "delete-function-url-config",
                "--function-name",
                &fn_name,
            ],
        )?;
        let _ = run_aws_allow_missing(
            region,
            &["lambda", "delete-function", "--function-name", &fn_name],
        )?;

        for role_name in [
            format!("{stack_name}-{suffix}-role"),
            format!("{stack_name}-{region}-{suffix}-role"),
        ] {
            if let Some(pols) = run_aws_allow_missing(
                region,
                &["iam", "list-role-policies", "--role-name", &role_name],
            )? {
                if let Some(names) = pols.get("PolicyNames").and_then(|v| v.as_array()) {
                    for p in names.iter().filter_map(|v| v.as_str()) {
                        let _ = run_aws_allow_missing(
                            region,
                            &[
                                "iam",
                                "delete-role-policy",
                                "--role-name",
                                &role_name,
                                "--policy-name",
                                p,
                            ],
                        )?;
                    }
                }
            }
            if let Some(attached) = run_aws_allow_missing(
                region,
                &[
                    "iam",
                    "list-attached-role-policies",
                    "--role-name",
                    &role_name,
                ],
            )? {
                if let Some(list) = attached.get("AttachedPolicies").and_then(|v| v.as_array()) {
                    for item in list {
                        if let Some(arn) = item.get("PolicyArn").and_then(|v| v.as_str()) {
                            let _ = run_aws_allow_missing(
                                region,
                                &[
                                    "iam",
                                    "detach-role-policy",
                                    "--role-name",
                                    &role_name,
                                    "--policy-arn",
                                    arn,
                                ],
                            )?;
                        }
                    }
                }
            }
            let _ =
                run_aws_allow_missing(region, &["iam", "delete-role", "--role-name", &role_name])?;
        }

        let log_group = format!("/aws/lambda/{stack_name}-{suffix}");
        let _ = run_aws_allow_missing(
            region,
            &["logs", "delete-log-group", "--log-group-name", &log_group],
        )?;
    }
    Ok(())
}

fn check_single_region_drift(
    creds: &CarbonAwsCredentials,
    environment: &str,
    region: &str,
    manifest_cell: Option<&CarbonCloudCellEntry>,
) -> CarbonDriftRegionResult {
    let stack_name = format!("jopedime-carbon-{environment}");
    let mut drift_reasons = Vec::new();
    let mut stack_status = None;
    let mut stack_release_id = None;
    let mut public_url = None;

    match read_stack_outputs_map(region, &stack_name) {
        Ok(Some(snap)) => {
            if !matches!(snap.status.as_str(), "CREATE_COMPLETE" | "UPDATE_COMPLETE") {
                drift_reasons.push(format!(
                    "unexpected CloudFormation StackStatus '{}'",
                    snap.status
                ));
            }
            stack_status = Some(snap.status);
            stack_release_id = snap.outputs.get("ReleaseId").cloned();
            public_url = snap
                .outputs
                .get("PublicEdgeEndpointUrl")
                .or_else(|| snap.outputs.get("EdgeEndpointUrl"))
                .cloned();
            if public_url.is_none() {
                drift_reasons
                    .push("stack missing PublicEdgeEndpointUrl/EdgeEndpointUrl output".to_owned());
            }
        }
        Ok(None) => {
            drift_reasons.push(format!(
                "CloudFormation stack '{stack_name}' not found in {region}"
            ));
        }
        Err(e) => {
            drift_reasons.push(format!("describe-stacks failed in {region}: {e}"));
        }
    }

    let live_versions = read_current_live_aliases(region, &stack_name);
    match &live_versions {
        Some(v) => {
            for (role, ver) in [
                ("edge", &v.edge_version),
                ("engine", &v.engine_version),
                ("eval", &v.eval_version),
            ] {
                if ver == "$LATEST" || ver.parse::<u64>().is_err() {
                    drift_reasons.push(format!(
                        "{role}:live points to non-immutable version '{ver}'"
                    ));
                }
            }
        }
        None => drift_reasons
            .push("missing one or more {edge,engine,eval}:live Lambda aliases".to_owned()),
    }

    let vault_kek_pubkey_b64 =
        match ensure_cell_secrets_in_vault(creds, region, environment, false, None, true) {
            Ok(v) if !v.secret_arn.is_empty() => Some(v.engine_kek_pubkey_b64),
            Ok(_) => {
                drift_reasons.push(format!("Secrets Manager vault missing in {region}"));
                None
            }
            Err(e) => {
                drift_reasons.push(format!("Secrets Manager read failed in {region}: {e}"));
                None
            }
        };

    let mut ping_kek_pubkey_b64 = None;
    let mut server_git_sha = None;
    if let Some(ref url) = public_url {
        match CarbonThinClient::connect_fastest(vec![url.clone()]) {
            Ok((client, probed)) => {
                if let Some(first) = probed.first() {
                    ping_kek_pubkey_b64 = first.engine_kek_pubkey_b64.clone();
                }
                if let Ok(Some(ver)) = client.fetch_server_version() {
                    server_git_sha = Some(ver.git_sha);
                }
            }
            Err(e) => drift_reasons.push(format!("live /v1/carbon/ping failed on {url}: {e}")),
        }
    }

    if let (Some(ref vk), Some(ref pk)) = (&vault_kek_pubkey_b64, &ping_kek_pubkey_b64) {
        if vk != pk {
            drift_reasons.push(format!(
                "KEK drift: Secrets Manager holds '{vk}', /v1/carbon/ping advertised '{pk}'"
            ));
        }
    }

    let mut manifest_kek_pubkey_b64 = None;
    if let Some(cell) = manifest_cell {
        let pin_opt = cell
            .regions
            .iter()
            .find(|r| r.region == region)
            .cloned()
            .or_else(|| {
                if cell.region == region {
                    Some(CarbonCloudRegionPin {
                        region: cell.region.clone(),
                        public_endpoint: cell.public_endpoint.clone(),
                        engine_kek_pubkey_b64: cell.engine_kek_pubkey_b64.clone(),
                        release_id: cell.release_id.clone(),
                    })
                } else {
                    None
                }
            });
        match pin_opt {
            Some(pin) => {
                manifest_kek_pubkey_b64 = pin.engine_kek_pubkey_b64.clone();
                if let Some(ref su) = public_url {
                    if su.trim_end_matches('/') != pin.public_endpoint.trim_end_matches('/') {
                        drift_reasons.push(format!(
                            "endpoint drift vs carbon-clouds.json: stack='{}', manifest='{}'",
                            su.trim_end_matches('/'),
                            pin.public_endpoint.trim_end_matches('/')
                        ));
                    }
                }
                if let (Some(ref mk), Some(ref pk)) =
                    (&manifest_kek_pubkey_b64, &ping_kek_pubkey_b64)
                {
                    if mk != pk {
                        drift_reasons.push(format!(
                            "KEK drift vs carbon-clouds.json: manifest='{mk}', ping='{pk}'"
                        ));
                    }
                }
            }
            None => drift_reasons.push(format!(
                "region '{region}' missing from carbon-clouds.json for cloud '{environment}'"
            )),
        }
    }

    CarbonDriftRegionResult {
        region: region.to_owned(),
        stack_name,
        stack_status,
        stack_release_id,
        public_edge_endpoint_url: public_url,
        live_versions,
        vault_kek_pubkey_b64,
        ping_kek_pubkey_b64,
        manifest_kek_pubkey_b64,
        server_git_sha,
        in_sync: drift_reasons.is_empty(),
        drift_reasons,
    }
}

/// Read-only multi-region drift verification comparing live CloudFormation stacks, Lambda `:live`
/// aliases, Secrets Manager KEKs, `/v1/carbon/ping`, and `carbon-clouds.json` (`--check-drift`).
pub fn execute_carbon_drift_check(
    environment: &str,
    raw_regions: &[String],
    clouds_manifest_override: Option<&str>,
) -> Result<CarbonDriftReport, String> {
    if !matches!(environment, "dev" | "staging" | "prod") {
        return Err(format!(
            "unsupported environment '{environment}' (expected dev, staging, or prod)"
        ));
    }
    let creds = CarbonAwsCredentials::from_env()?;
    let target_regions = expand_region_targets(raw_regions)?;
    let manifest_cell =
        crate::cli::client::resolve_public_cloud_endpoint(environment, clouds_manifest_override)
            .ok();

    let creds_ref = &creds;
    let mut results = Vec::with_capacity(target_regions.len());
    for chunk in target_regions.chunks(4) {
        let batch: Vec<CarbonDriftRegionResult> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunk.len());
            let cell_ref = manifest_cell.as_ref();
            for region in chunk {
                let reg = region.as_str();
                handles.push(s.spawn(move || {
                    check_single_region_drift(creds_ref, environment, reg, cell_ref)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| CarbonDriftRegionResult {
                        region: "unknown".to_owned(),
                        stack_name: format!("jopedime-carbon-{environment}"),
                        stack_status: None,
                        stack_release_id: None,
                        public_edge_endpoint_url: None,
                        live_versions: None,
                        vault_kek_pubkey_b64: None,
                        ping_kek_pubkey_b64: None,
                        manifest_kek_pubkey_b64: None,
                        server_git_sha: None,
                        in_sync: false,
                        drift_reasons: vec!["drift probe worker panicked".to_owned()],
                    })
                })
                .collect()
        });
        results.extend(batch);
    }

    let all_in_sync = results.iter().all(|r| r.in_sync);
    Ok(CarbonDriftReport {
        timestamp_epoch_ms: now_epoch_ms(),
        environment: environment.to_owned(),
        all_in_sync,
        regions: results,
    })
}

fn find_previous_published_version(
    region: &str,
    fn_name: &str,
    current: &str,
) -> Result<String, String> {
    let list = run_aws_json(
        region,
        &[
            "lambda",
            "list-versions-by-function",
            "--function-name",
            fn_name,
        ],
    )?;
    let mut numeric_versions: Vec<u64> = list
        .get("Versions")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("Version").and_then(|v| v.as_str()))
        .filter(|v| *v != "$LATEST" && *v != current)
        .filter_map(|v| v.parse::<u64>().ok())
        .collect();
    numeric_versions.sort_unstable();
    numeric_versions
        .pop()
        .map(|v| v.to_string())
        .ok_or_else(|| {
            format!("no prior retained version available on '{fn_name}' to roll back to")
        })
}

/// Atomically repoint `{edge,engine,eval}:live` Lambda aliases back to the prior retained immutable
/// versions (`< 1 s` zero-downtime traffic cutover) and verify `/v1/carbon/ping`.
pub fn execute_carbon_rollback(
    opts: &CarbonRollbackOptions,
) -> Result<CarbonRollbackReceipt, String> {
    let stack_name = format!("jopedime-carbon-{}", opts.environment);
    let target = match &opts.target_versions {
        Some(explicit) => explicit.clone(),
        None => {
            let current =
                read_current_live_aliases(&opts.region, &stack_name).ok_or_else(|| {
                    format!(
                        "unable to read current :live aliases for '{stack_name}' in {}",
                        opts.region
                    )
                })?;
            CarbonLiveAliasVersions {
                edge_version: find_previous_published_version(
                    &opts.region,
                    &format!("{stack_name}-edge"),
                    &current.edge_version,
                )?,
                engine_version: find_previous_published_version(
                    &opts.region,
                    &format!("{stack_name}-engine"),
                    &current.engine_version,
                )?,
                eval_version: find_previous_published_version(
                    &opts.region,
                    &format!("{stack_name}-eval"),
                    &current.eval_version,
                )?,
            }
        }
    };

    for (suffix, ver) in [
        ("eval", &target.eval_version),
        ("engine", &target.engine_version),
        ("edge", &target.edge_version),
    ] {
        let fn_name = format!("{stack_name}-{suffix}");
        let _ = run_aws_json(
            &opts.region,
            &[
                "lambda",
                "update-alias",
                "--function-name",
                &fn_name,
                "--name",
                "live",
                "--function-version",
                ver,
            ],
        )?;
    }

    let snap = read_stack_outputs_map(&opts.region, &stack_name)?
        .ok_or_else(|| format!("stack '{stack_name}' not found in {}", opts.region))?;
    let mut ping_verified = true;
    for key in ["EdgeEndpointUrl", "PublicEdgeEndpointUrl"] {
        if let Some(url) = snap.outputs.get(key) {
            let (_, probed) = CarbonThinClient::connect_fastest(vec![url.clone()])?;
            ping_verified = ping_verified && !probed.is_empty();
        }
    }

    Ok(CarbonRollbackReceipt {
        timestamp_epoch_ms: now_epoch_ms(),
        stack_name,
        region: opts.region.clone(),
        restored_versions: target,
        ping_verified,
    })
}

/// Execute atomic `:live` alias rollback across one or more regions (`canary`, `wave1`, `all-17`).
pub fn execute_carbon_rollback_multi(
    environment: &str,
    raw_regions: &[String],
    target_versions: Option<CarbonLiveAliasVersions>,
) -> Result<Vec<CarbonRollbackReceipt>, String> {
    let regions = expand_region_targets(raw_regions)?;
    let mut out = Vec::with_capacity(regions.len());
    for chunk in regions.chunks(4) {
        let batch: Vec<Result<CarbonRollbackReceipt, String>> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunk.len());
            for region in chunk {
                let reg = region.clone();
                let tv = target_versions.clone();
                handles.push(s.spawn(move || {
                    execute_carbon_rollback(&CarbonRollbackOptions {
                        environment: environment.to_owned(),
                        region: reg,
                        target_versions: tv,
                    })
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("regional rollback worker panicked".to_owned()))
                })
                .collect()
        });
        for item in batch {
            out.push(item?);
        }
    }
    Ok(out)
}

/// Query live CloudFormation stacks across `dev` and `prod` regions, probe `/v1/carbon/ping` on
/// every regional `PublicEdgeEndpointUrl`, verify regional `engine_kek_pubkey_b64` against AWS
/// Secrets Manager, and produce a signed-release-ready `jope-carbon-clouds-v2` manifest.
pub fn generate_public_clouds_manifest(
    dev_regions: &[String],
    prod_regions: &[String],
) -> Result<CarbonPublicCloudsManifest, String> {
    let creds = CarbonAwsCredentials::from_env()?;
    let mut clouds = BTreeMap::new();

    for (cloud_name, raw_list) in [("dev", dev_regions), ("prod", prod_regions)] {
        if raw_list.iter().all(|s| s.trim().is_empty()) {
            continue;
        }
        let expanded = expand_region_targets(raw_list)?;
        let stack_name = format!("jopedime-carbon-{cloud_name}");
        let mut pins = Vec::with_capacity(expanded.len());

        for region in &expanded {
            let snap = read_stack_outputs_map(region, &stack_name)?.ok_or_else(|| {
                format!("CloudFormation stack '{stack_name}' not found in region {region}")
            })?;
            let pub_url = snap
                .outputs
                .get("PublicEdgeEndpointUrl")
                .or_else(|| snap.outputs.get("EdgeEndpointUrl"))
                .cloned()
                .ok_or_else(|| {
                    format!("stack '{stack_name}' in {region} missing PublicEdgeEndpointUrl")
                })?;
            let release_id = snap.outputs.get("ReleaseId").cloned();
            let vault =
                ensure_cell_secrets_in_vault(&creds, region, cloud_name, false, None, true)?;
            let (_, probed) = CarbonThinClient::connect_fastest(vec![pub_url.clone()])?;
            let first = probed.first().ok_or_else(|| {
                format!("live /v1/carbon/ping probe returned empty for {pub_url} ({region})")
            })?;
            if first.engine_kek_pubkey_b64.as_deref() != Some(vault.engine_kek_pubkey_b64.as_str())
            {
                return Err(format!(
                    "live KEK mismatch on {cloud_name} ({region}): ping advertised {:?}, vault holds {}",
                    first.engine_kek_pubkey_b64, vault.engine_kek_pubkey_b64
                ));
            }
            pins.push(CarbonCloudRegionPin {
                region: region.clone(),
                public_endpoint: pub_url.trim_end_matches('/').to_owned(),
                engine_kek_pubkey_b64: Some(vault.engine_kek_pubkey_b64),
                release_id,
            });
        }

        let primary = pins.first().cloned().ok_or_else(|| {
            format!("no active regional endpoints discovered for cloud '{cloud_name}'")
        })?;
        clouds.insert(
            cloud_name.to_owned(),
            CarbonCloudCellEntry {
                environment: cloud_name.to_owned(),
                region: primary.region,
                public_endpoint: primary.public_endpoint,
                engine_kek_pubkey_b64: primary.engine_kek_pubkey_b64,
                release_id: primary.release_id,
                regions: pins,
            },
        );
    }

    Ok(CarbonPublicCloudsManifest {
        schema: "jope-carbon-clouds-v2".to_owned(),
        generated_epoch_ms: now_epoch_ms(),
        clouds,
    })
}
