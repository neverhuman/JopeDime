//! Pure-Rust Carbon release packager (`carbon package`), declarative CloudFormation ChangeSet
//! deployer (`carbon deploy`), Secrets Manager vault integration, and sub-second `:live` alias
//! rollback (`carbon rollback`).
//!
//! Compliant with `PYTHON_EXCEPTIONS.md` and `PY_TO_RUST_PLAN.md`: zero Python or shell wrapper
//! scripts, zero external `zip` binary dependencies, and zero plaintext secrets on disk or `argv`.

use crate::error::CarbonErrCtx as _;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use aws_lc_rs::rand as lc_rand;
use base64::Engine as _;
use flate2::{
    write::{DeflateEncoder, GzEncoder},
    Compression, Crc,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub use crate::cli::deploy_drift::{
    execute_carbon_drift_check, execute_carbon_rollback, execute_carbon_rollback_multi,
    generate_public_clouds_manifest, resolve_cell_stack_endpoint,
    resolve_cell_stack_public_endpoint, CarbonDriftRegionResult, CarbonDriftReport,
    CarbonRollbackOptions, CarbonRollbackReceipt,
};
use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::{
        client::CarbonThinClient,
        deploy_drift::{
            decommission_unstacked_prototype_resources, read_current_live_aliases,
            read_stack_outputs_map,
        },
        redteam::{execute_carbon_redteam_suite, execute_carbon_redteam_suite_with_tokens},
        runner::{execute_carbon_cli_run, CarbonCliRunOptions},
    },
    security::{
        auth::{CarbonAuthVerifier, CarbonCapabilityClaims},
        sigv4::{
            resolve_aws_cli_binary, resolve_s3_request_target, resolve_secretsmanager_endpoint,
            sign_aws_v4_headers_with_extra, CarbonAwsCredentials,
        },
        user_envelope::CarbonEngineKek,
    },
    storage::cas::sha256_lower_hex,
    tls::carbon_https_agent,
};

/// Canonical checked-in AWS CloudFormation template (`deploy/carbon/cloudformation-cell.json`).
pub const CLOUDFORMATION_CELL_TEMPLATE_JSON: &str =
    include_str!("../../../../deploy/carbon/cloudformation-cell.json");

/// Canonical 17 global Carbon serving regions (`opt-in-not-required` regions live in production AWS account)
/// ordered by wave promotion sequence.
pub const CANONICAL_17_REGIONS: &[&str] = &[
    "us-east-2",
    "us-east-1",
    "us-west-2",
    "us-west-1",
    "ca-central-1",
    "sa-east-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
];

const WAVE1_REGIONS: &[&str] = &[
    "us-east-2",
    "us-east-1",
    "us-west-2",
    "eu-central-1",
    "eu-west-1",
];

const DOS_EPOCH_MOD_TIME: u16 = 0x0000;
const DOS_EPOCH_MOD_DATE: u16 = 0x0021; // 1980-01-01
const UNIX_MODE_EXEC_755: u32 = 0o100755;
const UNIX_MODE_FILE_644: u32 = 0o100644;

/// `CarbonZipEntry` is a struct representing one deterministic in-memory file entry for a ZIP archive.
pub struct CarbonZipEntry {
    pub rel_path: String,
    pub bytes: Vec<u8>,
    pub unix_mode: u32,
}

/// `CarbonReleaseManifest` is a struct binding verified ARM64 Lambda artifacts, SHA-256 digests,
/// and the `r882` bundle digest for reproducible CloudFormation rollouts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonReleaseManifest {
    pub schema_version: u32,
    pub release_id: String,
    pub git_sha: String,
    pub r882_bundle_digest: String,
    pub edge_zip_file: String,
    pub edge_zip_bytes: u64,
    pub edge_zip_sha256_hex: String,
    pub edge_code_sha256_b64: String,
    pub worker_zip_file: String,
    pub worker_zip_bytes: u64,
    pub worker_zip_sha256_hex: String,
    pub worker_code_sha256_b64: String,
    pub r882_layer_zip_file: String,
    pub r882_layer_zip_bytes: u64,
    pub r882_layer_sha256_hex: String,
    pub r882_layer_sha256_b64: String,
}

/// `CarbonPackageOptions` is a struct configuring `carbon package`.
#[derive(Clone, Debug)]
pub struct CarbonPackageOptions {
    pub r882_dir: PathBuf,
    pub out_dir: PathBuf,
    pub edge_bin: Option<PathBuf>,
    pub worker_bin: Option<PathBuf>,
    pub build_jobs: usize,
    pub include_cli_binaries: bool,
}

/// `CarbonDeployOptions` is a struct configuring `carbon deploy`.
#[derive(Clone, Debug)]
pub struct CarbonDeployOptions {
    pub manifest_path: PathBuf,
    pub environment: String,
    pub regions: Vec<String>,
    pub artifacts_bucket_override: Option<String>,
    pub bootstrap_secrets: bool,
    pub decommission_unstacked_prototype: bool,
    pub dry_run: bool,
    pub skip_redteam: bool,
}

/// `CarbonLiveAliasVersions` is a struct recording active published Lambda version numbers for a cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonLiveAliasVersions {
    pub edge_version: String,
    pub engine_version: String,
    pub eval_version: String,
}

/// `CarbonRegionDeployReceipt` is a struct recording the outcome of deploying one regional Carbon cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonRegionDeployReceipt {
    pub region: String,
    pub stack_name: String,
    pub environment: String,
    pub release_id: String,
    pub dry_run: bool,
    pub change_set_action_count: usize,
    pub change_set_summary: Vec<String>,
    pub artifacts_bucket: String,
    pub cas_bucket: Option<String>,
    pub edge_endpoint_url: Option<String>,
    #[serde(default)]
    pub public_edge_endpoint_url: Option<String>,
    #[serde(default)]
    pub engine_kek_pubkey_b64: Option<String>,
    pub live_versions: Option<CarbonLiveAliasVersions>,
    pub previous_versions: Option<CarbonLiveAliasVersions>,
    pub r882_layer_version_arn: Option<String>,
    pub redteam_passed_vectors: Option<usize>,
    pub redteam_total_vectors: Option<usize>,
    pub wall_ms: f64,
}

/// `CarbonDeployReport` is a struct aggregating a single-region or multi-region rollout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonDeployReport {
    pub timestamp_epoch_ms: u64,
    pub environment: String,
    pub release_id: String,
    pub git_sha: String,
    pub dry_run: bool,
    pub regions: Vec<CarbonRegionDeployReceipt>,
}

#[derive(Serialize, Deserialize)]
struct CellSecretDocument {
    presign_secret_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_presign_secret_hex: Option<String>,
    auth_secret_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_auth_secret_hex: Option<String>,
    seal_secret_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_seal_secret_hex: Option<String>,
    internal_token: String,
    engine_kek_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_engine_kek_hex: Option<String>,
    engine_kek_pubkey_b64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    app_auth_pubkeys_b64: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct EdgeSecretDocument {
    presign_secret_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_presign_secret_hex: Option<String>,
    auth_secret_hex: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_auth_secret_hex: Option<String>,
    engine_kek_pubkey_b64: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    app_auth_pubkeys_b64: Option<String>,
}

pub(crate) struct VaultSecretBinding {
    pub(crate) secret_arn: String,
    version_id: String,
    edge_secret_arn: String,
    edge_version_id: String,
    pub(crate) engine_kek_pubkey_b64: String,
    pub(crate) auth_secret_hex: Zeroizing<String>,
    pub(crate) app_auth_pubkeys_b64: Option<String>,
}

/// Expand region preset tokens (`canary`, `wave1`, `all-17`) or validate explicit region names.
pub fn expand_region_targets(raw: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for item in raw {
        let token = item.trim();
        if token.is_empty() {
            continue;
        }
        match token {
            "canary" => out.push("us-east-2".to_owned()),
            "wave1" => {
                for &r in WAVE1_REGIONS {
                    if !out.iter().any(|existing| existing == r) {
                        out.push(r.to_owned());
                    }
                }
            }
            "all-17" | "global" => {
                for &r in CANONICAL_17_REGIONS {
                    if !out.iter().any(|existing| existing == r) {
                        out.push(r.to_owned());
                    }
                }
            }
            explicit => {
                if !explicit
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                {
                    return Err(format!("invalid AWS region identifier '{explicit}'"));
                }
                if !out.iter().any(|existing| existing == explicit) {
                    out.push(explicit.to_owned());
                }
            }
        }
    }
    if out.is_empty() {
        out.push("us-east-2".to_owned());
    }
    Ok(out)
}

/// Build a deterministic, bit-for-bit reproducible PKZIP (`2.0`) archive in memory using `flate2`.
pub fn build_deterministic_zip_bytes(mut entries: Vec<CarbonZipEntry>) -> Result<Vec<u8>, String> {
    if entries.len() > usize::from(u16::MAX) {
        return Err(format!("too many ZIP entries ({})", entries.len()));
    }
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let mut archive = Vec::new();
    let mut central_dir = Vec::new();

    for entry in &entries {
        let rel = entry.rel_path.trim_start_matches('/');
        if rel.is_empty() || rel.contains('\\') || rel.contains("..") {
            return Err(format!("unsafe ZIP entry path '{}'", entry.rel_path));
        }
        let name_bytes = rel.as_bytes();
        if name_bytes.len() > usize::from(u16::MAX) {
            return Err(format!("ZIP entry path too long: '{rel}'"));
        }
        if u64::try_from(entry.bytes.len()).unwrap_or(u64::MAX) > u64::from(u32::MAX) {
            return Err(format!("ZIP entry '{rel}' exceeds 4 GiB PKZIP 2.0 bound"));
        }

        let mut crc = Crc::new();
        crc.update(&entry.bytes);
        let crc32 = crc.sum();

        let (method, compressed) = if entry.bytes.is_empty() {
            (0u16, Vec::new())
        } else {
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&entry.bytes)
                .map_err(|e| format!("DEFLATE compress '{rel}': {e}"))?;
            let comp = enc
                .finish()
                .map_err(|e| format!("DEFLATE finish '{rel}': {e}"))?;
            (8u16, comp)
        };

        let local_offset = u32::try_from(archive.len())
            .map_err(|_| "ZIP archive exceeds 4 GiB offset bound".to_owned())?;
        let comp_len = u32::try_from(compressed.len())
            .map_err(|_| "compressed entry exceeds 4 GiB".to_owned())?;
        let uncomp_len = entry.bytes.len() as u32;
        let name_len = name_bytes.len() as u16;

        // Local file header (30 bytes + filename)
        archive.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        archive.extend_from_slice(&20u16.to_le_bytes());
        archive.extend_from_slice(&0u16.to_le_bytes());
        archive.extend_from_slice(&method.to_le_bytes());
        archive.extend_from_slice(&DOS_EPOCH_MOD_TIME.to_le_bytes());
        archive.extend_from_slice(&DOS_EPOCH_MOD_DATE.to_le_bytes());
        archive.extend_from_slice(&crc32.to_le_bytes());
        archive.extend_from_slice(&comp_len.to_le_bytes());
        archive.extend_from_slice(&uncomp_len.to_le_bytes());
        archive.extend_from_slice(&name_len.to_le_bytes());
        archive.extend_from_slice(&0u16.to_le_bytes());
        archive.extend_from_slice(name_bytes);
        archive.extend_from_slice(&compressed);

        // Central directory header (46 bytes + filename)
        let version_made_by: u16 = (3u16 << 8) | 20u16; // Unix + PKZIP 2.0
        let ext_attrs: u32 = entry.unix_mode << 16;
        central_dir.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central_dir.extend_from_slice(&version_made_by.to_le_bytes());
        central_dir.extend_from_slice(&20u16.to_le_bytes());
        central_dir.extend_from_slice(&0u16.to_le_bytes());
        central_dir.extend_from_slice(&method.to_le_bytes());
        central_dir.extend_from_slice(&DOS_EPOCH_MOD_TIME.to_le_bytes());
        central_dir.extend_from_slice(&DOS_EPOCH_MOD_DATE.to_le_bytes());
        central_dir.extend_from_slice(&crc32.to_le_bytes());
        central_dir.extend_from_slice(&comp_len.to_le_bytes());
        central_dir.extend_from_slice(&uncomp_len.to_le_bytes());
        central_dir.extend_from_slice(&name_len.to_le_bytes());
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // extra len
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // comment len
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // disk start
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        central_dir.extend_from_slice(&ext_attrs.to_le_bytes());
        central_dir.extend_from_slice(&local_offset.to_le_bytes());
        central_dir.extend_from_slice(name_bytes);
    }

    let cd_offset = u32::try_from(archive.len())
        .map_err(|_| "ZIP central directory offset exceeds 4 GiB".to_owned())?;
    let cd_size = u32::try_from(central_dir.len())
        .map_err(|_| "ZIP central directory size exceeds 4 GiB".to_owned())?;
    let count = entries.len() as u16;

    archive.extend_from_slice(&central_dir);
    archive.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive.extend_from_slice(&count.to_le_bytes());
    archive.extend_from_slice(&count.to_le_bytes());
    archive.extend_from_slice(&cd_size.to_le_bytes());
    archive.extend_from_slice(&cd_offset.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());

    Ok(archive)
}

/// Verify that a 64-bit little-endian Linux ELF executable is statically linked (zero `PT_INTERP`
/// dynamic loader segment), preventing accidental glibc linkage in Lambda or musl CLI artifacts (`DEP-SEC-03`).
pub fn verify_static_linux_elf(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 64 || &bytes[0..4] != b"\x7fELF" {
        return Err("binary is not a valid 64-bit ELF executable".to_owned());
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        return Err("ELF binary must be 64-bit little-endian".to_owned());
    }
    let e_phoff = u64::from_le_bytes([
        bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
    ]) as usize;
    let e_phentsize = u16::from_le_bytes([bytes[54], bytes[55]]) as usize;
    let e_phnum = u16::from_le_bytes([bytes[56], bytes[57]]) as usize;
    if e_phnum > 0 && e_phentsize >= 56 {
        let end = e_phoff.saturating_add(e_phnum.saturating_mul(e_phentsize));
        if end > bytes.len() {
            return Err("ELF program header table exceeds file length".to_owned());
        }
        for idx in 0..e_phnum {
            let off = e_phoff + idx * e_phentsize;
            let p_type =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            // PT_INTERP == 3 indicates a dynamically linked ELF requiring ld-linux.so
            if p_type == 3 {
                return Err(
                    "ELF binary contains PT_INTERP dynamic loader segment (must be statically linked musl)"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

/// Verify an ELF64 `aarch64` (`EM_AARCH64 == 0xB7`) binary and optionally enforce `CT3` edge
/// isolation (zero embedded `r882` / `jope-service` / `feat-math` symbols).
pub fn verify_arm64_lambda_binary(
    bytes: &[u8],
    enforce_ct3_edge_fence: bool,
) -> Result<(), String> {
    verify_static_linux_elf(bytes)?;
    let e_machine = u16::from_le_bytes([bytes[18], bytes[19]]);
    if e_machine != 0x00B7 {
        return Err(format!(
            "Lambda bootstrap ELF machine mismatch: expected EM_AARCH64 (0x00B7), got 0x{e_machine:04X}"
        ));
    }
    if enforce_ct3_edge_fence {
        for forbidden in [
            b"models/bundles/r882".as_slice(),
            b"ServingAdmissionSlate".as_slice(),
            b"CarbonInProcessExecutor".as_slice(),
        ] {
            if bytes.windows(forbidden.len()).any(|w| w == forbidden) {
                return Err(format!(
                    "CT3 violation: carbon-edge binary contains forbidden engine/neural symbol '{}'",
                    String::from_utf8_lossy(forbidden)
                ));
            }
        }
    }
    Ok(())
}

/// Synthesize a release-bound CloudFormation template JSON by rewriting immutable
/// `{Edge,Engine,Eval}VersionCurrent` logical IDs to `{Edge,Engine,Eval}Version<Suffix>`.
pub fn synthesize_cell_template_for_release(release_id: &str) -> Result<String, String> {
    synthesize_cell_template_for_release_with_edge_tag(release_id, None)
}

/// Synthesize a release-bound CloudFormation template JSON with an optional 8-char alphanumeric
/// `edge_cfg_tag` appended to `EdgeVersion` so updating `edge-secrets` / `AppAuthPubkeysB64` on an
/// existing `release_id` publishes a fresh immutable `AWS::Lambda::Version` and recycles `:live`.
pub fn synthesize_cell_template_for_release_with_edge_tag(
    release_id: &str,
    edge_cfg_tag: Option<&str>,
) -> Result<String, String> {
    let suffix: String = release_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(24)
        .collect();
    if suffix.len() < 6 {
        return Err(format!(
            "release_id '{release_id}' must contain at least 6 ASCII alphanumeric characters"
        ));
    }
    let edge_extra: String = edge_cfg_tag
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();

    let mut root: serde_json::Value = serde_json::from_str(CLOUDFORMATION_CELL_TEMPLATE_JSON)
        .err_ctx("parse embedded cloudformation-cell.json")?;

    let edge_ver_id = format!("EdgeVersion{suffix}{edge_extra}");
    let engine_ver_id = format!("EngineVersion{suffix}");
    let eval_ver_id = format!("EvalVersion{suffix}");

    {
        let resources = root
            .get_mut("Resources")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| "cloudformation-cell.json missing Resources object".to_owned())?;

        for (old_key, new_key, alias_key) in [
            ("EdgeVersionCurrent", &edge_ver_id, "EdgeLiveAlias"),
            ("EngineVersionCurrent", &engine_ver_id, "EngineLiveAlias"),
            ("EvalVersionCurrent", &eval_ver_id, "EvalLiveAlias"),
        ] {
            let ver_obj = resources
                .remove(old_key)
                .ok_or_else(|| format!("missing logical resource {old_key}"))?;
            resources.insert(new_key.clone(), ver_obj);

            let alias = resources
                .get_mut(alias_key)
                .and_then(|a| a.get_mut("Properties"))
                .and_then(|p| p.as_object_mut())
                .ok_or_else(|| format!("missing Properties on {alias_key}"))?;
            alias.insert(
                "FunctionVersion".to_owned(),
                serde_json::json!({"Fn::GetAtt": [new_key, "Version"]}),
            );
        }
    }

    {
        let outputs = root
            .get_mut("Outputs")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| "cloudformation-cell.json missing Outputs object".to_owned())?;
        for (out_key, new_ver_id) in [
            ("EdgePublishedVersion", &edge_ver_id),
            ("EnginePublishedVersion", &engine_ver_id),
            ("EvalPublishedVersion", &eval_ver_id),
        ] {
            if let Some(entry) = outputs.get_mut(out_key).and_then(|v| v.as_object_mut()) {
                entry.insert(
                    "Value".to_owned(),
                    serde_json::json!({"Fn::GetAtt": [new_ver_id, "Version"]}),
                );
            }
        }
    }

    serde_json::to_string_pretty(&root).err_ctx("serialize synthesized CloudFormation template")
}

fn collect_files_recursively(
    base_dir: &Path,
    cur_dir: &Path,
    zip_prefix: &str,
    out: &mut Vec<CarbonZipEntry>,
) -> Result<(), String> {
    let mut dir_entries: Vec<_> = fs::read_dir(cur_dir)
        .map_err(|e| format!("read_dir {}: {e}", cur_dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("iter read_dir {}: {e}", cur_dir.display()))?;
    dir_entries.sort_by_key(|e| e.file_name());

    for entry in dir_entries {
        let path = entry.path();
        let meta =
            fs::symlink_metadata(&path).map_err(|e| format!("stat {}: {e}", path.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "refusing symlink inside r882 bundle directory: {}",
                path.display()
            ));
        }
        if meta.is_dir() {
            collect_files_recursively(base_dir, &path, zip_prefix, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(base_dir)
                .map_err(|e| format!("strip_prefix {}: {e}", path.display()))?;
            let rel_slash = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            out.push(CarbonZipEntry {
                rel_path: format!("{}/{rel_slash}", zip_prefix.trim_end_matches('/')),
                bytes,
                unix_mode: UNIX_MODE_FILE_644,
            });
        }
    }
    Ok(())
}

fn resolve_git_head_sha() -> String {
    for env_key in ["JOPE_BUILD_SOURCE_SHA", "JOPE_SOURCE_SHA"] {
        if let Ok(raw) = std::env::var(env_key) {
            let trimmed = raw.trim();
            if trimmed.len() == 40 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
                return trimmed.to_ascii_lowercase();
            }
        }
    }
    if let Ok(out) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
            if s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()) {
                return s.to_ascii_lowercase();
            }
        }
    }
    crate::CARBON_GIT_SHA.to_owned()
}

struct ZigMuslWrappers {
    cc: PathBuf,
    cxx: PathBuf,
    ar: PathBuf,
    linker: PathBuf,
}

fn find_executable_in_path(bin_name: &str) -> Option<PathBuf> {
    let path_os = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_os) {
        let candidate = dir.join(bin_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn ensure_zig_musl_wrappers_for_arch(
    wrapper_dir: &Path,
    zig_target: &str,
    rust_target: &str,
) -> Result<Option<ZigMuslWrappers>, String> {
    let zig_candidates = [
        std::env::var_os("CARBON_ZIG_BIN").map(PathBuf::from),
        find_executable_in_path("zig"),
        Some(PathBuf::from(
            "target/zig-stage/zig-linux-x86_64-0.13.0/zig",
        )),
        crate::home_path::absolute_home_path(
            std::env::var_os("HOME").as_deref(),
            &[".local", "zig", "zig"],
        ),
    ];
    let Some(zig_bin) = zig_candidates.into_iter().flatten().find(|p| p.is_file()) else {
        return Ok(None);
    };
    let zig_abs = fs::canonicalize(&zig_bin)
        .map_err(|e| format!("canonicalize zig binary {}: {e}", zig_bin.display()))?;
    let cache_dir = wrapper_dir.join("zig-cache");
    let local_cache_dir = cache_dir.join("local");
    fs::create_dir_all(&local_cache_dir)
        .map_err(|e| format!("create zig cache dir {}: {e}", local_cache_dir.display()))?;
    let cache_abs = fs::canonicalize(&cache_dir).map_err(crate::error::err_to_string)?;
    let local_cache_abs =
        fs::canonicalize(&local_cache_dir).map_err(crate::error::err_to_string)?;

    let cc_script = wrapper_dir.join(format!("{zig_target}-cc"));
    let cxx_script = wrapper_dir.join(format!("{zig_target}-cxx"));
    let ar_script = wrapper_dir.join(format!("{zig_target}-ar"));
    let link_script = wrapper_dir.join(format!("{zig_target}-link"));

    let cc_body = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport ZIG_GLOBAL_CACHE_DIR=\"{}\"\nexport ZIG_LOCAL_CACHE_DIR=\"{}\"\nargs=()\nfor arg in \"$@\"; do\n  case \"$arg\" in\n    --target={rust_target}|-lgcc_s) ;;\n    *) args+=(\"$arg\") ;;\n  esac\ndone\nexec \"{}\" cc -target {zig_target} \"${{args[@]}}\"\n",
        cache_abs.display(),
        local_cache_abs.display(),
        zig_abs.display()
    );
    let cxx_body = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport ZIG_GLOBAL_CACHE_DIR=\"{}\"\nexport ZIG_LOCAL_CACHE_DIR=\"{}\"\nargs=()\nfor arg in \"$@\"; do\n  case \"$arg\" in\n    --target={rust_target}|-lgcc_s) ;;\n    *) args+=(\"$arg\") ;;\n  esac\ndone\nexec \"{}\" c++ -target {zig_target} \"${{args[@]}}\"\n",
        cache_abs.display(),
        local_cache_abs.display(),
        zig_abs.display()
    );
    let ar_body = format!(
        "#!/usr/bin/env bash\nexport ZIG_GLOBAL_CACHE_DIR=\"{}\"\nexport ZIG_LOCAL_CACHE_DIR=\"{}\"\nexec \"{}\" ar \"$@\"\n",
        cache_abs.display(),
        local_cache_abs.display(),
        zig_abs.display()
    );
    let link_body = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport ZIG_GLOBAL_CACHE_DIR=\"{}\"\nexport ZIG_LOCAL_CACHE_DIR=\"{}\"\nargs=()\nfor arg in \"$@\"; do\n  case \"$arg\" in\n    --target={rust_target}|-lgcc_s) ;;\n    */self-contained/*crt*.o) ;;\n    *) args+=(\"$arg\") ;;\n  esac\ndone\nexec \"{}\" cc -target {zig_target} \"${{args[@]}}\"\n",
        cache_abs.display(),
        local_cache_abs.display(),
        zig_abs.display()
    );

    for (path, body) in [
        (&cc_script, cc_body),
        (&cxx_script, cxx_body),
        (&ar_script, ar_body),
        (&link_script, link_body),
    ] {
        fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(path)
                .map_err(|e| format!("stat {}: {e}", path.display()))?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms)
                .map_err(|e| format!("chmod {}: {e}", path.display()))?;
        }
    }
    Ok(Some(ZigMuslWrappers {
        cc: fs::canonicalize(&cc_script).map_err(crate::error::err_to_string)?,
        cxx: fs::canonicalize(&cxx_script).map_err(crate::error::err_to_string)?,
        ar: fs::canonicalize(&ar_script).map_err(crate::error::err_to_string)?,
        linker: fs::canonicalize(&link_script).map_err(crate::error::err_to_string)?,
    }))
}

fn ensure_zig_musl_wrappers(wrapper_dir: &Path) -> Result<Option<ZigMuslWrappers>, String> {
    ensure_zig_musl_wrappers_for_arch(
        wrapper_dir,
        "aarch64-linux-musl",
        "aarch64-unknown-linux-musl",
    )
}

fn cross_compile_musl_thin_cli_binary(
    out_dir: &Path,
    jobs: usize,
    zig_target: &str,
    rust_target: &str,
) -> Result<PathBuf, String> {
    let wrapper_dir = out_dir.join(".zig-toolchain");
    let build_tmp = wrapper_dir.join("build-tmp");
    fs::create_dir_all(&build_tmp)
        .map_err(|e| format!("create build-tmp {}: {e}", build_tmp.display()))?;
    let build_tmp_abs = fs::canonicalize(&build_tmp).map_err(crate::error::err_to_string)?;
    let zig_wrappers = ensure_zig_musl_wrappers_for_arch(&wrapper_dir, zig_target, rust_target)?;
    let jobs_str = jobs.max(1).to_string();
    let snake = rust_target.replace('-', "_");
    let upper = snake.to_ascii_uppercase();
    let mut cmd = Command::new("cargo");
    cmd.args([
        "build",
        "--locked",
        "--profile",
        "carbon-cli",
        "--target",
        rust_target,
        "-p",
        "jope-carbon",
        "--no-default-features",
        "--features",
        "cli",
        "--bin",
        "carbon",
        "-j",
        &jobs_str,
    ])
    .env("CARGO_INCREMENTAL", "0")
    .env("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    .env("AWS_LC_SYS_CMAKE_BUILDER", "0")
    .env("TMPDIR", &build_tmp_abs);
    if let Some(w) = &zig_wrappers {
        cmd.env(format!("CC_{snake}"), &w.cc)
            .env(format!("CXX_{snake}"), &w.cxx)
            .env(format!("AR_{snake}"), &w.ar)
            .env(format!("CARGO_TARGET_{upper}_LINKER"), &w.linker);
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn cargo build for {rust_target} thin carbon: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cargo build for {rust_target} thin carbon exited with {status}"
        ));
    }
    let meta_out = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .err_ctx("cargo metadata failed")?;
    if !meta_out.status.success() {
        return Err("cargo metadata failed when resolving target_directory".to_owned());
    }
    let meta_json: serde_json::Value =
        serde_json::from_slice(&meta_out.stdout).err_ctx("parse cargo metadata JSON")?;
    let target_root = meta_json
        .get("target_directory")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "cargo metadata missing target_directory".to_owned())?;
    Ok(Path::new(target_root)
        .join(rust_target)
        .join("carbon-cli")
        .join("carbon"))
}

/// Build a deterministic POSIX USTAR `.tar` archive in memory containing a single `0755` executable entry.
pub fn build_deterministic_single_bin_tar(
    entry_name: &str,
    bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let name_bytes = entry_name.as_bytes();
    if name_bytes.is_empty() || name_bytes.len() > 99 || entry_name.contains('/') {
        return Err(format!("invalid USTAR binary entry name '{entry_name}'"));
    }
    let mut header = [0u8; 512];
    header[0..name_bytes.len()].copy_from_slice(name_bytes);
    header[100..108].copy_from_slice(b"0000755\0");
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    let size_octal = format!("{:011o}\0", bytes.len());
    header[124..136].copy_from_slice(size_octal.as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[148..156].copy_from_slice(b"        ");
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let chksum: u32 = header.iter().map(|&b| u32::from(b)).sum();
    let chk_octal = format!("{chksum:06o}\0 ");
    header[148..156].copy_from_slice(chk_octal.as_bytes());

    let pad_len = (512 - (bytes.len() % 512)) % 512;
    let mut tar = Vec::with_capacity(512 + bytes.len() + pad_len + 1_024);
    tar.extend_from_slice(&header);
    tar.extend_from_slice(bytes);
    if pad_len > 0 {
        tar.resize(tar.len() + pad_len, 0u8);
    }
    tar.extend_from_slice(&[0u8; 1_024]);
    Ok(tar)
}

/// Build a deterministic POSIX USTAR `.tar.gz` containing a single `0755` executable entry.
pub fn build_deterministic_single_bin_targz(
    entry_name: &str,
    bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let tar = build_deterministic_single_bin_tar(entry_name, bytes)?;
    let mut enc = GzEncoder::new(Vec::new(), Compression::best());
    enc.write_all(&tar).err_ctx("gzip write tar stream")?;
    enc.finish().err_ctx("gzip finish tar.gz")
}

/// Pipe `input` into an external stream compressor/decompressor (`xz`, `zstd`, etc.) while
/// concurrently draining `stdout` so compressed outputs exceeding the OS pipe buffer (~64 KiB)
/// never deadlock on `pipe_write`.
pub fn try_external_stream_compress(tool: &str, args: &[&str], input: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let stdin = child.stdin.take();
    let (write_res, wait_res) = std::thread::scope(|s| {
        let writer = s.spawn(move || -> std::io::Result<()> {
            if let Some(mut pipe) = stdin {
                pipe.write_all(input)?;
                pipe.flush()?;
            }
            Ok(())
        });
        let out = child.wait_with_output();
        let write_res = writer
            .join()
            .unwrap_or_else(|_| Err(std::io::Error::other("stdin writer thread panicked")));
        (write_res, out)
    });
    if write_res.is_err() {
        return None;
    }
    let out = wait_res.ok()?;
    if out.status.success() && !out.stdout.is_empty() {
        Some(out.stdout)
    } else {
        None
    }
}

fn cross_compile_arm64_binaries(out_dir: &Path, jobs: usize) -> Result<(PathBuf, PathBuf), String> {
    let wrapper_dir = out_dir.join(".zig-toolchain");
    let build_tmp = wrapper_dir.join("build-tmp");
    fs::create_dir_all(&build_tmp)
        .map_err(|e| format!("create build-tmp {}: {e}", build_tmp.display()))?;
    let build_tmp_abs = fs::canonicalize(&build_tmp).map_err(crate::error::err_to_string)?;
    let zig_wrappers = ensure_zig_musl_wrappers(&wrapper_dir)?;
    let jobs_str = jobs.max(1).to_string();

    for (label, extra_args) in [
        (
            "carbon-edge (CT3 isolated, carbon-cli profile)",
            vec![
                "build",
                "--locked",
                "--profile",
                "carbon-cli",
                "--target",
                "aarch64-unknown-linux-musl",
                "-p",
                "jope-carbon",
                "--no-default-features",
                "--features",
                "edge",
                "--bin",
                "carbon-edge",
                "-j",
                &jobs_str,
            ],
        ),
        (
            "carbon-worker (stripped dist engine+eval)",
            vec![
                "build",
                "--locked",
                "--profile",
                "dist",
                "--target",
                "aarch64-unknown-linux-musl",
                "-p",
                "jope-carbon",
                "--no-default-features",
                "--features",
                "eval",
                "--bin",
                "carbon-worker",
                "-j",
                &jobs_str,
            ],
        ),
    ] {
        let mut cmd = Command::new("cargo");
        cmd.args(&extra_args)
            .env("CARGO_INCREMENTAL", "0")
            .env("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
            .env("AWS_LC_SYS_CMAKE_BUILDER", "0")
            .env("TMPDIR", &build_tmp_abs);
        if let Some(w) = &zig_wrappers {
            cmd.env("CC_aarch64_unknown_linux_musl", &w.cc)
                .env("CXX_aarch64_unknown_linux_musl", &w.cxx)
                .env("AR_aarch64_unknown_linux_musl", &w.ar)
                .env("CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER", &w.linker);
        }
        let status = cmd
            .status()
            .map_err(|e| format!("failed to spawn cargo build for {label}: {e}"))?;
        if !status.success() {
            return Err(format!(
                "cargo cross-compile for {label} exited with {status}"
            ));
        }
    }

    // Locate target directory via cargo metadata (supports workspace and worktrees cleanly)
    let meta_out = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .err_ctx("cargo metadata failed")?;
    if !meta_out.status.success() {
        return Err("cargo metadata failed when resolving target_directory".to_owned());
    }
    let meta_json: serde_json::Value =
        serde_json::from_slice(&meta_out.stdout).err_ctx("parse cargo metadata JSON")?;
    let target_root = meta_json
        .get("target_directory")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "cargo metadata missing target_directory".to_owned())?;
    let arch_root = Path::new(target_root).join("aarch64-unknown-linux-musl");
    Ok((
        arch_root.join("carbon-cli").join("carbon-edge"),
        arch_root.join("dist").join("carbon-worker"),
    ))
}

fn rust_target_is_installed(rust_target: &str) -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .is_some_and(|list| list.lines().any(|line| line.trim() == rust_target))
}

fn emit_cli_binary_and_archives(
    out_dir: &Path,
    base_name: &str,
    raw_bin: &[u8],
    sum_lines: &mut Vec<String>,
) -> Result<(), String> {
    let bin_out_path = out_dir.join(base_name);
    fs::write(&bin_out_path, raw_bin).map_err(|e| format!("write {base_name}: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&bin_out_path, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("chmod 0755 {base_name}: {e}"))?;
    }
    sum_lines.push(format!("{}  {base_name}", sha256_lower_hex(raw_bin)));

    let tar_bytes = build_deterministic_single_bin_tar("carbon", raw_bin)?;
    let tgz = build_deterministic_single_bin_targz("carbon", raw_bin)?;
    let tgz_name = format!("{base_name}.tar.gz");
    fs::write(out_dir.join(&tgz_name), &tgz).map_err(|e| format!("write {tgz_name}: {e}"))?;
    sum_lines.push(format!("{}  {tgz_name}", sha256_lower_hex(&tgz)));

    if let Some(tzst) =
        try_external_stream_compress("zstd", &["-19", "--ultra", "-q", "-c"], &tar_bytes)
    {
        let tzst_name = format!("{base_name}.tar.zst");
        fs::write(out_dir.join(&tzst_name), &tzst)
            .map_err(|e| format!("write {tzst_name}: {e}"))?;
        sum_lines.push(format!("{}  {tzst_name}", sha256_lower_hex(&tzst)));
    }
    if let Some(txz) = try_external_stream_compress("xz", &["-9e", "-q", "-c"], &tar_bytes) {
        let txz_name = format!("{base_name}.tar.xz");
        fs::write(out_dir.join(&txz_name), &txz).map_err(|e| format!("write {txz_name}: {e}"))?;
        sum_lines.push(format!("{}  {txz_name}", sha256_lower_hex(&txz)));
    }
    Ok(())
}

/// Package and verify ARM64 Lambda artifacts (`carbon-edge.zip`, `carbon-worker.zip`,
/// `r882-layer.zip`, and `release-manifest.json`).
pub fn execute_carbon_package(
    opts: &CarbonPackageOptions,
) -> Result<CarbonReleaseManifest, String> {
    // 1. Cryptographically verify the r882 release bundle before packaging a single byte!
    let r882_bundle_digest =
        jope_service::discover::r#loop::carbon::verify_r882_release_dir(&opts.r882_dir)
            .map_err(|e| format!("r882 verification failed: {e:#}"))?;
    assemble_verified_carbon_package(opts, r882_bundle_digest)
}

/// Assemble deterministic ARM64 Lambda archives (`carbon-edge.zip`, `carbon-worker.zip`,
/// `r882-layer.zip`) and `release-manifest.json` after `r882_bundle_digest` has been verified.
pub fn assemble_verified_carbon_package(
    opts: &CarbonPackageOptions,
    r882_bundle_digest: String,
) -> Result<CarbonReleaseManifest, String> {
    fs::create_dir_all(&opts.out_dir)
        .map_err(|e| format!("create package out_dir {}: {e}", opts.out_dir.display()))?;

    // 2. Resolve or cross-compile ARM64 musl binaries
    let (edge_bin_path, worker_bin_path) = match (&opts.edge_bin, &opts.worker_bin) {
        (Some(e), Some(w)) => (e.clone(), w.clone()),
        (None, None) => cross_compile_arm64_binaries(&opts.out_dir, opts.build_jobs)?,
        _ => {
            return Err(
                "provide both --edge-bin and --worker-bin or omit both to cross-compile".to_owned(),
            )
        }
    };

    let edge_bytes =
        fs::read(&edge_bin_path).map_err(|e| format!("read {}: {e}", edge_bin_path.display()))?;
    let worker_bytes = fs::read(&worker_bin_path)
        .map_err(|e| format!("read {}: {e}", worker_bin_path.display()))?;

    verify_arm64_lambda_binary(&edge_bytes, true)?;
    verify_arm64_lambda_binary(&worker_bytes, false)?;

    // 3. Build deterministic reproducible ZIPs (+ optional multi-arch thin musl CLI archives)
    let edge_zip = build_deterministic_zip_bytes(vec![CarbonZipEntry {
        rel_path: "bootstrap".to_owned(),
        bytes: edge_bytes,
        unix_mode: UNIX_MODE_EXEC_755,
    }])?;
    let worker_zip = build_deterministic_zip_bytes(vec![CarbonZipEntry {
        rel_path: "bootstrap".to_owned(),
        bytes: worker_bytes,
        unix_mode: UNIX_MODE_EXEC_755,
    }])?;

    let mut layer_entries = Vec::new();
    collect_files_recursively(
        &opts.r882_dir,
        &opts.r882_dir,
        "jope-models/r882",
        &mut layer_entries,
    )?;
    let layer_zip = build_deterministic_zip_bytes(layer_entries)?;

    if opts.include_cli_binaries {
        let x64_bin_path = cross_compile_musl_thin_cli_binary(
            &opts.out_dir,
            opts.build_jobs,
            "x86_64-linux-musl",
            "x86_64-unknown-linux-musl",
        )?;
        let arm_bin_path = cross_compile_musl_thin_cli_binary(
            &opts.out_dir,
            opts.build_jobs,
            "aarch64-linux-musl",
            "aarch64-unknown-linux-musl",
        )?;
        let x64_bytes =
            fs::read(&x64_bin_path).map_err(|e| format!("read {}: {e}", x64_bin_path.display()))?;
        let arm_bytes =
            fs::read(&arm_bin_path).map_err(|e| format!("read {}: {e}", arm_bin_path.display()))?;
        verify_static_linux_elf(&x64_bytes)?;
        verify_arm64_lambda_binary(&arm_bytes, true)?;

        let mut sum_lines = Vec::new();
        for (base_name, raw_bin) in [
            ("carbon-linux-x86_64-musl", x64_bytes.as_slice()),
            ("carbon-linux-aarch64-musl", arm_bytes.as_slice()),
        ] {
            emit_cli_binary_and_archives(&opts.out_dir, base_name, raw_bin, &mut sum_lines)?;
        }
        if rust_target_is_installed("armv7-unknown-linux-musleabihf") {
            if let Ok(armv7_path) = cross_compile_musl_thin_cli_binary(
                &opts.out_dir,
                opts.build_jobs,
                "arm-linux-musleabihf",
                "armv7-unknown-linux-musleabihf",
            ) {
                if let Ok(armv7_bytes) = fs::read(&armv7_path) {
                    emit_cli_binary_and_archives(
                        &opts.out_dir,
                        "carbon-linux-armv7-musleabihf",
                        &armv7_bytes,
                        &mut sum_lines,
                    )?;
                }
            }
        }
        let sums = format!("{}\n", sum_lines.join("\n"));
        fs::write(opts.out_dir.join("CLI-SHA256SUMS"), sums).err_ctx("write CLI-SHA256SUMS")?;
    }

    // Clean up temporary zig wrapper dir if created inside out_dir
    let _ = fs::remove_dir_all(opts.out_dir.join(".zig-toolchain"));

    let edge_sha_raw = Sha256::digest(&edge_zip);
    let worker_sha_raw = Sha256::digest(&worker_zip);
    let layer_sha_raw = Sha256::digest(&layer_zip);

    let edge_sha_hex = hex::encode(edge_sha_raw);
    let worker_sha_hex = hex::encode(worker_sha_raw);
    let layer_sha_hex = hex::encode(layer_sha_raw);

    let edge_b64 = base64::engine::general_purpose::STANDARD.encode(edge_sha_raw);
    let worker_b64 = base64::engine::general_purpose::STANDARD.encode(worker_sha_raw);
    let layer_b64 = base64::engine::general_purpose::STANDARD.encode(layer_sha_raw);

    let git_sha = resolve_git_head_sha();
    let composite_hash = sha256_lower_hex(
        format!("{git_sha}:{edge_sha_hex}:{worker_sha_hex}:{layer_sha_hex}").as_bytes(),
    );
    let release_id = format!("{}{}", &git_sha[..8], &composite_hash[..8]);

    let edge_file = "carbon-edge.zip".to_owned();
    let worker_file = "carbon-worker.zip".to_owned();
    let layer_file = "r882-layer.zip".to_owned();

    fs::write(opts.out_dir.join(&edge_file), &edge_zip)
        .map_err(|e| format!("write {edge_file}: {e}"))?;
    fs::write(opts.out_dir.join(&worker_file), &worker_zip)
        .map_err(|e| format!("write {worker_file}: {e}"))?;
    fs::write(opts.out_dir.join(&layer_file), &layer_zip)
        .map_err(|e| format!("write {layer_file}: {e}"))?;

    let manifest = CarbonReleaseManifest {
        schema_version: 1,
        release_id,
        git_sha,
        r882_bundle_digest,
        edge_zip_file: edge_file,
        edge_zip_bytes: edge_zip.len() as u64,
        edge_zip_sha256_hex: edge_sha_hex,
        edge_code_sha256_b64: edge_b64,
        worker_zip_file: worker_file,
        worker_zip_bytes: worker_zip.len() as u64,
        worker_zip_sha256_hex: worker_sha_hex,
        worker_code_sha256_b64: worker_b64,
        r882_layer_zip_file: layer_file,
        r882_layer_zip_bytes: layer_zip.len() as u64,
        r882_layer_sha256_hex: layer_sha_hex,
        r882_layer_sha256_b64: layer_b64,
    };

    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).err_ctx("encode release-manifest.json")?;
    fs::write(opts.out_dir.join("release-manifest.json"), manifest_bytes)
        .err_ctx("write release-manifest.json")?;

    Ok(manifest)
}

pub(crate) fn run_aws_json(region: &str, args: &[&str]) -> Result<serde_json::Value, String> {
    let aws_bin = resolve_aws_cli_binary();
    let out = Command::new(&aws_bin)
        .arg("--region")
        .arg(region)
        .args(args)
        .arg("--output")
        .arg("json")
        .output()
        .map_err(|e| format!("invoke aws {:?}: {e}", args.first()))?;
    if !out.status.success() {
        return Err(format!(
            "aws {} failed ({}): {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    if out.stdout.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_slice(&out.stdout).err_ctx("decode aws JSON output")
}

pub(crate) fn run_aws_allow_missing(
    region: &str,
    args: &[&str],
) -> Result<Option<serde_json::Value>, String> {
    let aws_bin = resolve_aws_cli_binary();
    let out = Command::new(&aws_bin)
        .arg("--region")
        .arg(region)
        .args(args)
        .arg("--output")
        .arg("json")
        .output()
        .map_err(|e| format!("invoke aws {:?}: {e}", args.first()))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("does not exist")
            || stderr.contains("ResourceNotFoundException")
            || stderr.contains("NoSuchEntity")
            || stderr.contains("NotFound")
            || stderr.contains("404")
        {
            return Ok(None);
        }
        return Err(format!(
            "aws {} failed ({}): {}",
            args.join(" "),
            out.status,
            stderr.trim()
        ));
    }
    if out.stdout.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(Some(serde_json::Value::Null));
    }
    let val = serde_json::from_slice(&out.stdout).err_ctx("decode aws JSON output")?;
    Ok(Some(val))
}

fn call_secrets_manager_rpc(
    creds: &CarbonAwsCredentials,
    region: &str,
    target_op: &str,
    payload: &[u8],
) -> Result<(u16, Zeroizing<Vec<u8>>), String> {
    let (host, url) = resolve_secretsmanager_endpoint(region)?;
    let full_target = format!("secretsmanager.{target_op}");
    let extra = [
        ("content-type", "application/x-amz-json-1.1"),
        ("x-amz-target", full_target.as_str()),
    ];
    let headers = sign_aws_v4_headers_with_extra(
        creds,
        region,
        "secretsmanager",
        "POST",
        &host,
        "/",
        "",
        payload,
        now_epoch_ms() / 1_000,
        &extra,
    );
    let mut req = carbon_https_agent()
        .post(&url)
        .config()
        .http_status_as_error(false)
        .build();
    for (k, v) in &headers {
        req = req.header(k, v);
    }
    let mut resp = req
        .send(payload)
        .map_err(|e| format!("SecretsManager {target_op} TLS request failed: {e}"))?;
    let status = resp.status().as_u16();
    let raw_vec = crate::storage::cas::read_bounded(
        resp.body_mut().as_reader(),
        4 * 1024 * 1024,
        &format!("SecretsManager {target_op} response"),
    )?;
    Ok((status, Zeroizing::new(raw_vec)))
}

fn random_32byte_hex() -> Result<Zeroizing<String>, String> {
    let mut raw = Zeroizing::new([0u8; 32]);
    lc_rand::fill(&mut *raw).map_err(|_| "CSPRNG failure generating cell secret".to_owned())?;
    Ok(Zeroizing::new(hex::encode(*raw)))
}

fn generate_cell_secret_document(
    auth_override_hex: Option<&str>,
) -> Result<(Zeroizing<String>, String, Zeroizing<String>), String> {
    let presign = random_32byte_hex()?;
    let auth = match auth_override_hex {
        Some(explicit) => Zeroizing::new(explicit.trim().to_owned()),
        None => random_32byte_hex()?,
    };
    let seal = random_32byte_hex()?;
    let internal = random_32byte_hex()?;
    let mut kek_scalar = Zeroizing::new([0u8; 32]);
    let kek = loop {
        lc_rand::fill(&mut *kek_scalar)
            .map_err(|_| "CSPRNG failure generating P-256 KEK scalar".to_owned())?;
        if let Ok(loaded) = CarbonEngineKek::from_scalar_bytes(&kek_scalar) {
            break loaded;
        }
    };
    let pub_b64 = kek.pubkey_b64();
    let kek_hex = Zeroizing::new(hex::encode(*kek_scalar));

    let doc = CellSecretDocument {
        presign_secret_hex: (*presign).clone(),
        previous_presign_secret_hex: None,
        auth_secret_hex: (*auth).clone(),
        previous_auth_secret_hex: None,
        seal_secret_hex: (*seal).clone(),
        previous_seal_secret_hex: None,
        internal_token: (*internal).clone(),
        engine_kek_hex: (*kek_hex).clone(),
        previous_engine_kek_hex: None,
        engine_kek_pubkey_b64: pub_b64.clone(),
        app_auth_pubkeys_b64: None,
    };
    let serialized =
        Zeroizing::new(serde_json::to_string(&doc).err_ctx("serialize CellSecretDocument")?);
    Ok((serialized, pub_b64, auth))
}

fn format_uuid_from_random_hex(tok_hex: &str) -> String {
    format!(
        "{}-{}-{}-{}-{}",
        &tok_hex[0..8],
        &tok_hex[8..12],
        &tok_hex[12..16],
        &tok_hex[16..20],
        &tok_hex[20..32]
    )
}

fn ensure_edge_secrets_in_vault(
    creds: &CarbonAwsCredentials,
    region: &str,
    environment: &str,
    cell_doc: &CellSecretDocument,
    dry_run: bool,
) -> Result<(String, String), String> {
    let secret_id = format!("jopedime/carbon/{environment}/edge-secrets");
    let desired = EdgeSecretDocument {
        presign_secret_hex: cell_doc.presign_secret_hex.clone(),
        previous_presign_secret_hex: cell_doc.previous_presign_secret_hex.clone(),
        auth_secret_hex: cell_doc.auth_secret_hex.clone(),
        previous_auth_secret_hex: cell_doc.previous_auth_secret_hex.clone(),
        engine_kek_pubkey_b64: cell_doc.engine_kek_pubkey_b64.clone(),
        app_auth_pubkeys_b64: cell_doc.app_auth_pubkeys_b64.clone(),
    };
    let desired_json =
        Zeroizing::new(serde_json::to_string(&desired).err_ctx("serialize EdgeSecretDocument")?);

    let get_req = serde_json::to_vec(&serde_json::json!({
        "SecretId": secret_id,
        "VersionStage": "AWSCURRENT",
    }))
    .map_err(crate::error::err_to_string)?;
    let (status, body) = call_secrets_manager_rpc(creds, region, "GetSecretValue", &get_req)?;
    if status == 200 {
        let val: serde_json::Value =
            serde_json::from_slice(&body).err_ctx("parse EdgeSecrets GetSecretValue response")?;
        let arn = val
            .get("ARN")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "EdgeSecrets response missing ARN".to_owned())?
            .to_owned();
        let mut version_id = val
            .get("VersionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "EdgeSecrets response missing VersionId".to_owned())?
            .to_owned();
        let cur_str = Zeroizing::new(
            val.get("SecretString")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned(),
        );
        let in_sync = serde_json::from_str::<EdgeSecretDocument>(&cur_str)
            .map(|cur| {
                cur.presign_secret_hex == desired.presign_secret_hex
                    && cur.previous_presign_secret_hex == desired.previous_presign_secret_hex
                    && cur.auth_secret_hex == desired.auth_secret_hex
                    && cur.previous_auth_secret_hex == desired.previous_auth_secret_hex
                    && cur.engine_kek_pubkey_b64 == desired.engine_kek_pubkey_b64
                    && cur.app_auth_pubkeys_b64 == desired.app_auth_pubkeys_b64
            })
            .unwrap_or(false);
        if !in_sync && !dry_run {
            let tok_hex = random_32byte_hex()?;
            let put_payload = Zeroizing::new(
                serde_json::to_vec(&serde_json::json!({
                    "SecretId": secret_id,
                    "ClientRequestToken": format_uuid_from_random_hex(&tok_hex),
                    "SecretString": &*desired_json,
                }))
                .map_err(crate::error::err_to_string)?,
            );
            let (p_status, p_body) =
                call_secrets_manager_rpc(creds, region, "PutSecretValue", &put_payload)?;
            if p_status != 200 {
                return Err(format!(
                    "SecretsManager PutSecretValue '{secret_id}' in {region} failed (HTTP {p_status})"
                ));
            }
            let put_val: serde_json::Value = serde_json::from_slice(&p_body)
                .err_ctx("parse EdgeSecrets PutSecretValue response")?;
            version_id = put_val
                .get("VersionId")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "EdgeSecrets PutSecretValue missing VersionId".to_owned())?
                .to_owned();
        }
        return Ok((arn, version_id));
    }

    let body_text = String::from_utf8_lossy(&body);
    if !body_text.contains("ResourceNotFoundException") {
        return Err(format!(
            "SecretsManager GetSecretValue '{secret_id}' failed (HTTP {status})"
        ));
    }
    if dry_run {
        return Ok((String::new(), String::new()));
    }

    let tok_hex = random_32byte_hex()?;
    let create_payload = Zeroizing::new(
        serde_json::to_vec(&serde_json::json!({
            "Name": secret_id,
            "ClientRequestToken": format_uuid_from_random_hex(&tok_hex),
            "Description": format!("JopeDime Carbon {environment} isolated edge secrets (zero engine KEK / seal keys)"),
            "SecretString": &*desired_json,
        }))
        .map_err(crate::error::err_to_string)?,
    );
    let (c_status, c_body) =
        call_secrets_manager_rpc(creds, region, "CreateSecret", &create_payload)?;
    if c_status != 200 {
        return Err(format!(
            "SecretsManager CreateSecret '{secret_id}' failed (HTTP {c_status})"
        ));
    }
    let created: serde_json::Value =
        serde_json::from_slice(&c_body).err_ctx("parse EdgeSecrets CreateSecret response")?;
    let arn = created
        .get("ARN")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "EdgeSecrets CreateSecret missing ARN".to_owned())?
        .to_owned();
    let version_id = created
        .get("VersionId")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "EdgeSecrets CreateSecret missing VersionId".to_owned())?
        .to_owned();
    Ok((arn, version_id))
}

fn normalize_app_pubkeys_override(raw: Option<&str>) -> Result<Option<Option<String>>, String> {
    let Some(val) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if val.eq_ignore_ascii_case("CLEAR") {
        return Ok(Some(None));
    }
    let _ = jope_envelope::parse_pinned_pubkeys_csv(val)
        .map_err(|e| format!("invalid --app-auth-pubkeys-b64: {e}"))?;
    Ok(Some(Some(val.to_owned())))
}

pub(crate) fn ensure_cell_secrets_in_vault(
    creds: &CarbonAwsCredentials,
    region: &str,
    environment: &str,
    bootstrap_if_missing: bool,
    required_auth_secret_hex: Option<&str>,
    dry_run: bool,
) -> Result<VaultSecretBinding, String> {
    ensure_cell_secrets_in_vault_with_pubkeys(
        creds,
        region,
        environment,
        bootstrap_if_missing,
        required_auth_secret_hex,
        None,
        dry_run,
    )
}

pub(crate) fn ensure_cell_secrets_in_vault_with_pubkeys(
    creds: &CarbonAwsCredentials,
    region: &str,
    environment: &str,
    bootstrap_if_missing: bool,
    required_auth_secret_hex: Option<&str>,
    app_auth_pubkeys_override: Option<&str>,
    dry_run: bool,
) -> Result<VaultSecretBinding, String> {
    let desired_app_pubkeys = normalize_app_pubkeys_override(app_auth_pubkeys_override)?;
    let secret_id = format!("jopedime/carbon/{environment}/cell-secrets");
    let get_req = serde_json::to_vec(&serde_json::json!({
        "SecretId": secret_id,
        "VersionStage": "AWSCURRENT",
    }))
    .map_err(crate::error::err_to_string)?;

    let (status, body) = call_secrets_manager_rpc(creds, region, "GetSecretValue", &get_req)?;
    if status == 200 {
        let val: serde_json::Value = serde_json::from_slice(&body)
            .err_ctx("parse SecretsManager GetSecretValue response")?;
        let arn = val
            .get("ARN")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "SecretsManager response missing ARN".to_owned())?
            .to_owned();
        let mut version_id = val
            .get("VersionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "SecretsManager response missing VersionId".to_owned())?
            .to_owned();
        let secret_str = Zeroizing::new(
            val.get("SecretString")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "SecretsManager response missing SecretString".to_owned())?
                .to_owned(),
        );
        let mut parsed_doc: CellSecretDocument = serde_json::from_str(&secret_str)
            .map_err(|e| format!("invalid CellSecretDocument schema in '{secret_id}': {e}"))?;
        let mut mutated_cell_doc = false;
        if let Some(req_auth) = required_auth_secret_hex {
            if parsed_doc.auth_secret_hex != req_auth.trim() {
                let prior_auth =
                    std::mem::replace(&mut parsed_doc.auth_secret_hex, req_auth.trim().to_owned());
                if !prior_auth.trim().is_empty() {
                    parsed_doc.previous_auth_secret_hex = Some(prior_auth);
                }
                mutated_cell_doc = true;
            }
        }
        if let Some(target_pubkeys) = desired_app_pubkeys {
            if parsed_doc.app_auth_pubkeys_b64 != target_pubkeys {
                parsed_doc.app_auth_pubkeys_b64 = target_pubkeys;
                mutated_cell_doc = true;
            }
        }
        if mutated_cell_doc && !dry_run {
            let updated_json = Zeroizing::new(
                serde_json::to_string(&parsed_doc)
                    .err_ctx("serialize synced CellSecretDocument")?,
            );
            let tok_hex = random_32byte_hex()?;
            let put_payload = Zeroizing::new(
                serde_json::to_vec(&serde_json::json!({
                    "SecretId": secret_id,
                    "ClientRequestToken": format_uuid_from_random_hex(&tok_hex),
                    "SecretString": &*updated_json,
                }))
                .map_err(crate::error::err_to_string)?,
            );
            let (p_status, p_body) =
                call_secrets_manager_rpc(creds, region, "PutSecretValue", &put_payload)?;
            if p_status != 200 {
                return Err(format!(
                    "SecretsManager PutSecretValue '{secret_id}' in {region} failed (HTTP {p_status})"
                ));
            }
            let put_val: serde_json::Value =
                serde_json::from_slice(&p_body).err_ctx("parse PutSecretValue response")?;
            version_id = put_val
                .get("VersionId")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "PutSecretValue response missing VersionId".to_owned())?
                .to_owned();
        }
        let (edge_arn, edge_vid) =
            ensure_edge_secrets_in_vault(creds, region, environment, &parsed_doc, dry_run)?;
        return Ok(VaultSecretBinding {
            secret_arn: arn,
            version_id,
            edge_secret_arn: edge_arn,
            edge_version_id: edge_vid,
            engine_kek_pubkey_b64: parsed_doc.engine_kek_pubkey_b64,
            auth_secret_hex: Zeroizing::new(parsed_doc.auth_secret_hex),
            app_auth_pubkeys_b64: parsed_doc.app_auth_pubkeys_b64,
        });
    }

    let body_text = String::from_utf8_lossy(&body);
    if !body_text.contains("ResourceNotFoundException") {
        return Err(format!(
            "SecretsManager GetSecretValue '{secret_id}' failed (HTTP {status})"
        ));
    }
    if !bootstrap_if_missing {
        return Err(format!(
            "secret '{secret_id}' does not exist in {region} (re-run with --bootstrap-secrets to initialize)"
        ));
    }
    if dry_run {
        let (_, pubkey_b64, auth_hex) = generate_cell_secret_document(required_auth_secret_hex)?;
        return Ok(VaultSecretBinding {
            secret_arn: String::new(),
            version_id: String::new(),
            edge_secret_arn: String::new(),
            edge_version_id: String::new(),
            engine_kek_pubkey_b64: pubkey_b64,
            auth_secret_hex: auth_hex,
            app_auth_pubkeys_b64: desired_app_pubkeys.flatten(),
        });
    }

    let (secret_json, pubkey_b64, auth_hex) =
        generate_cell_secret_document(required_auth_secret_hex)?;
    let mut parsed_doc: CellSecretDocument =
        serde_json::from_str(&secret_json).err_ctx("parse generated CellSecretDocument")?;
    if let Some(target_pubkeys) = desired_app_pubkeys {
        parsed_doc.app_auth_pubkeys_b64 = target_pubkeys;
    }
    let secret_json =
        Zeroizing::new(serde_json::to_string(&parsed_doc).err_ctx("serialize CellSecretDocument")?);
    let tok_hex = random_32byte_hex()?;
    let client_req_token = format_uuid_from_random_hex(&tok_hex);
    let create_payload = Zeroizing::new(
        serde_json::to_vec(&serde_json::json!({
            "Name": secret_id,
            "ClientRequestToken": client_req_token,
            "Description": format!("JopeDime Carbon {environment} cell cryptographic secrets (managed by carbon deploy)"),
            "SecretString": &*secret_json,
        }))
        .map_err(crate::error::err_to_string)?,
    );
    let (c_status, c_body) =
        call_secrets_manager_rpc(creds, region, "CreateSecret", &create_payload)?;
    if c_status != 200 {
        return Err(format!(
            "SecretsManager CreateSecret '{secret_id}' failed (HTTP {c_status})"
        ));
    }
    let created: serde_json::Value =
        serde_json::from_slice(&c_body).err_ctx("parse CreateSecret response")?;
    let arn = created
        .get("ARN")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "CreateSecret response missing ARN".to_owned())?
        .to_owned();
    let version_id = created
        .get("VersionId")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "CreateSecret response missing VersionId".to_owned())?
        .to_owned();
    let (edge_arn, edge_vid) =
        ensure_edge_secrets_in_vault(creds, region, environment, &parsed_doc, false)?;

    Ok(VaultSecretBinding {
        secret_arn: arn,
        version_id,
        edge_secret_arn: edge_arn,
        edge_version_id: edge_vid,
        engine_kek_pubkey_b64: pubkey_b64,
        auth_secret_hex: auth_hex,
        app_auth_pubkeys_b64: parsed_doc.app_auth_pubkeys_b64,
    })
}

/// Mint a scoped `ctk1.*` capability token directly in `Zeroizing` RAM from AWS Secrets Manager
/// (`jopedime/carbon/<env>/cell-secrets`) without exposing raw `auth_secret_hex` to shell env or disk.
pub fn mint_capability_token_from_vault(
    environment: &str,
    region: &str,
    tenant_id: &str,
    tier: jope_api::carbon::CarbonTierWire,
    ttl_secs: u64,
    admin: bool,
) -> Result<String, String> {
    let max_ttl = crate::security::auth::MAX_MINT_TOKEN_TTL_SECS;
    if ttl_secs == 0 || ttl_secs > max_ttl {
        return Err(format!(
            "ttl_secs must be in 1..={max_ttl} (90 days max), got {ttl_secs}"
        ));
    }
    let creds = CarbonAwsCredentials::from_env()?;
    let secret_id = format!("jopedime/carbon/{environment}/cell-secrets");
    let get_req = serde_json::to_vec(&serde_json::json!({
        "SecretId": secret_id,
        "VersionStage": "AWSCURRENT",
    }))
    .map_err(crate::error::err_to_string)?;
    let (status, body) = call_secrets_manager_rpc(&creds, region, "GetSecretValue", &get_req)?;
    if status != 200 {
        return Err(format!(
            "SecretsManager GetSecretValue '{secret_id}' failed (HTTP {status}): {}",
            String::from_utf8_lossy(&body)
        ));
    }
    let val: serde_json::Value = serde_json::from_slice(&body).err_ctx("parse secret JSON")?;
    let secret_str = Zeroizing::new(
        val.get("SecretString")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing SecretString".to_owned())?
            .to_owned(),
    );
    let doc: CellSecretDocument =
        serde_json::from_str(&secret_str).err_ctx("parse secret schema")?;
    let auth_hex = Zeroizing::new(doc.auth_secret_hex);
    let verifier = CarbonAuthVerifier::from_hex(&auth_hex).map_err(crate::error::err_to_string)?;
    let exp = (now_epoch_ms() / 1_000).saturating_add(ttl_secs);
    verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: tenant_id.to_owned(),
            tier,
            exp_epoch_secs: exp,
            admin,
        })
        .map_err(crate::error::err_to_string)
}

fn ensure_artifacts_bucket(region: &str, bucket: &str, dry_run: bool) -> Result<bool, String> {
    let exists =
        run_aws_allow_missing(region, &["s3api", "head-bucket", "--bucket", bucket])?.is_some();
    if dry_run {
        return Ok(exists);
    }
    if !exists {
        if region == "us-east-1" {
            let _ = run_aws_json(region, &["s3api", "create-bucket", "--bucket", bucket])?;
        } else {
            let cfg = format!("LocationConstraint={region}");
            let _ = run_aws_json(
                region,
                &[
                    "s3api",
                    "create-bucket",
                    "--bucket",
                    bucket,
                    "--create-bucket-configuration",
                    &cfg,
                ],
            )?;
        }
    }
    let _ = run_aws_json(
        region,
        &[
            "s3api",
            "put-public-access-block",
            "--bucket",
            bucket,
            "--public-access-block-configuration",
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true",
        ],
    )?;
    let _ = run_aws_json(
        region,
        &[
            "s3api",
            "put-bucket-encryption",
            "--bucket",
            bucket,
            "--server-side-encryption-configuration",
            r#"{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}"#,
        ],
    )?;
    let _ = run_aws_json(
        region,
        &[
            "s3api",
            "put-bucket-versioning",
            "--bucket",
            bucket,
            "--versioning-configuration",
            "Status=Enabled",
        ],
    )?;
    let _ = run_aws_json(
        region,
        &[
            "s3api",
            "put-bucket-lifecycle-configuration",
            "--bucket",
            bucket,
            "--lifecycle-configuration",
            r#"{"Rules":[{"ID":"AbortIncompleteMultipartUploads","Status":"Enabled","Filter":{"Prefix":""},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":1}}]}"#,
        ],
    )?;
    let partition = if region.starts_with("us-gov-") {
        "aws-us-gov"
    } else if region.starts_with("cn-") {
        "aws-cn"
    } else {
        "aws"
    };
    let policy_doc = serde_json::json!({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DenyInsecureTransport",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                format!("arn:{partition}:s3:::{bucket}"),
                format!("arn:{partition}:s3:::{bucket}/*"),
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }]
    })
    .to_string();
    let _ = run_aws_json(
        region,
        &[
            "s3api",
            "put-bucket-policy",
            "--bucket",
            bucket,
            "--policy",
            &policy_doc,
        ],
    )?;
    Ok(true)
}

fn s3_stage_object_verified(
    creds: &CarbonAwsCredentials,
    region: &str,
    bucket: &str,
    key: &str,
    local_file: &Path,
    expected_sha256_hex: &str,
    dry_run: bool,
) -> Result<bool, String> {
    let bytes = fs::read(local_file).map_err(|e| format!("read {}: {e}", local_file.display()))?;
    let digest = Sha256::digest(&bytes);
    let actual_sha = hex::encode(digest);
    if actual_sha != expected_sha256_hex.to_ascii_lowercase() {
        return Err(format!(
            "artifact {} SHA-256 mismatch against release-manifest.json",
            local_file.display()
        ));
    }
    let expected_sha_b64 = base64::engine::general_purpose::STANDARD.encode(digest);

    let (host, uri, url) = resolve_s3_request_target(bucket, region, key).map_err(str::to_owned)?;

    // Check HEAD first (with x-amz-checksum-mode: ENABLED) so content-addressed re-deploys verify SHA-256!
    let head_headers = sign_aws_v4_headers_with_extra(
        creds,
        region,
        "s3",
        "HEAD",
        &host,
        &uri,
        "",
        b"",
        now_epoch_ms() / 1_000,
        &[("x-amz-checksum-mode", "ENABLED")],
    );
    let mut head_req = carbon_https_agent()
        .head(&url)
        .config()
        .http_status_as_error(false)
        .build();
    for (k, v) in &head_headers {
        head_req = head_req.header(k, v);
    }
    if let Ok(resp) = head_req.call() {
        if resp.status().as_u16() == 200 {
            let len_ok = resp
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<usize>().ok())
                == Some(bytes.len());
            let checksum_ok = resp
                .headers()
                .get("x-amz-checksum-sha256")
                .and_then(|v| v.to_str().ok())
                .map(|hdr| hdr.trim() == expected_sha_b64)
                .unwrap_or(true);
            if len_ok && checksum_ok {
                return Ok(true);
            }
        }
    }

    if dry_run {
        return Ok(false);
    }

    let put_headers = sign_aws_v4_headers_with_extra(
        creds,
        region,
        "s3",
        "PUT",
        &host,
        &uri,
        "",
        &bytes,
        now_epoch_ms() / 1_000,
        &[
            ("content-type", "application/zip"),
            ("x-amz-checksum-sha256", expected_sha_b64.as_str()),
        ],
    );
    let mut put_req = carbon_https_agent()
        .put(&url)
        .config()
        .http_status_as_error(false)
        .build();
    for (k, v) in &put_headers {
        put_req = put_req.header(k, v);
    }
    let mut resp = put_req
        .send(&bytes)
        .map_err(|e| format!("S3 PUT s3://{bucket}/{key} failed: {e}"))?;
    let status = resp.status().as_u16();
    if status >= 300 {
        let err_body = crate::storage::cas::read_bounded(
            resp.body_mut().as_reader(),
            1024 * 1024,
            "S3 PUT error body",
        )
        .unwrap_or_default();
        return Err(format!(
            "S3 PUT s3://{bucket}/{key} HTTP {status}: {}",
            String::from_utf8_lossy(&err_body)
        ));
    }
    Ok(true)
}

fn wait_for_change_set(
    region: &str,
    stack_name: &str,
    change_set_name: &str,
) -> Result<(bool, Vec<String>), String> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        let desc = run_aws_json(
            region,
            &[
                "cloudformation",
                "describe-change-set",
                "--stack-name",
                stack_name,
                "--change-set-name",
                change_set_name,
            ],
        )?;
        let status = desc
            .get("Status")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let reason = desc
            .get("StatusReason")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if status == "CREATE_COMPLETE" {
            let mut summary = Vec::new();
            if let Some(changes) = desc.get("Changes").and_then(|v| v.as_array()) {
                for ch in changes {
                    if let Some(rc) = ch.get("ResourceChange") {
                        let action = rc.get("Action").and_then(|v| v.as_str()).unwrap_or("?");
                        let logical = rc
                            .get("LogicalResourceId")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?");
                        let rtype = rc
                            .get("ResourceType")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?");
                        summary.push(format!("{action} {logical} ({rtype})"));
                    }
                }
            }
            return Ok((true, summary));
        }
        if status == "FAILED" {
            if reason.contains("didn't contain changes")
                || reason.contains("No updates are to be performed")
            {
                return Ok((false, Vec::new()));
            }
            return Err(format!(
                "ChangeSet '{change_set_name}' on '{stack_name}' failed: {reason}"
            ));
        }
        thread::sleep(Duration::from_secs(2));
    }
    Err(format!(
        "timed out waiting for ChangeSet '{change_set_name}' on '{stack_name}'"
    ))
}

fn wait_for_stack_completion(region: &str, stack_name: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(600);
    while Instant::now() < deadline {
        let Some(snap) = read_stack_outputs_map(region, stack_name)? else {
            return Err(format!("stack '{stack_name}' disappeared during rollout"));
        };
        match snap.status.as_str() {
            "CREATE_COMPLETE" | "UPDATE_COMPLETE" => return Ok(()),
            s if s.ends_with("_IN_PROGRESS") => {
                thread::sleep(Duration::from_secs(4));
            }
            failed => {
                // Fetch latest failure event reason for clear diagnostics
                let events = run_aws_allow_missing(
                    region,
                    &[
                        "cloudformation",
                        "describe-stack-events",
                        "--stack-name",
                        stack_name,
                        "--max-items",
                        "15",
                    ],
                )?;
                let mut reasons = Vec::new();
                if let Some(ev_arr) = events
                    .as_ref()
                    .and_then(|v| v.get("StackEvents"))
                    .and_then(|v| v.as_array())
                {
                    for ev in ev_arr {
                        let st = ev
                            .get("ResourceStatus")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        if st.contains("FAILED") {
                            let lid = ev
                                .get("LogicalResourceId")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?");
                            let rsn = ev
                                .get("ResourceStatusReason")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?");
                            reasons.push(format!("{lid}: {rsn}"));
                        }
                    }
                }
                return Err(format!(
                    "CloudFormation stack '{stack_name}' entered terminal failure state {failed}: {}",
                    reasons.join(" | ")
                ));
            }
        }
    }
    Err(format!(
        "timed out waiting for CloudFormation stack '{stack_name}' in {region}"
    ))
}

#[derive(Clone, Copy)]
struct RegionDeployContext<'a> {
    creds: &'a CarbonAwsCredentials,
    account_id: &'a str,
    manifest: &'a CarbonReleaseManifest,
    manifest_dir: &'a Path,
    opts: &'a CarbonDeployOptions,
    global_auth_secret_hex: &'a str,
    app_auth_pubkeys_override: Option<&'a str>,
}

fn deploy_single_region(
    ctx: &RegionDeployContext<'_>,
    region: &str,
) -> Result<CarbonRegionDeployReceipt, String> {
    let RegionDeployContext {
        creds,
        account_id,
        manifest,
        manifest_dir,
        opts,
        global_auth_secret_hex,
        app_auth_pubkeys_override,
    } = *ctx;
    let t0 = Instant::now();
    let stack_name = format!("jopedime-carbon-{}", opts.environment);
    let artifacts_bucket = opts
        .artifacts_bucket_override
        .clone()
        .unwrap_or_else(|| format!("jopedime-carbon-artifacts-{account_id}-{region}"));

    // 1. Ensure regional secrets exist in AWS Secrets Manager with global `auth_secret_hex` sync,
    //    dedicated `edge-secrets` (`DEP-SEC-01`), and strictly read-only `--dry-run` (`DEP-SEC-02`)!
    let vault = ensure_cell_secrets_in_vault_with_pubkeys(
        creds,
        region,
        &opts.environment,
        opts.bootstrap_secrets,
        Some(global_auth_secret_hex),
        app_auth_pubkeys_override,
        opts.dry_run,
    )?;

    // 2. Stage content-addressed release ZIPs to regional S3 bucket (read-only HEAD check in --dry-run)
    let bucket_exists = ensure_artifacts_bucket(region, &artifacts_bucket, opts.dry_run)?;
    let edge_key = format!(
        "releases/{}/{}",
        manifest.release_id, manifest.edge_zip_file
    );
    let worker_key = format!(
        "releases/{}/{}",
        manifest.release_id, manifest.worker_zip_file
    );
    let layer_key = format!(
        "releases/{}/{}",
        manifest.release_id, manifest.r882_layer_zip_file
    );

    let edge_staged = s3_stage_object_verified(
        creds,
        region,
        &artifacts_bucket,
        &edge_key,
        &manifest_dir.join(&manifest.edge_zip_file),
        &manifest.edge_zip_sha256_hex,
        opts.dry_run,
    )?;
    let worker_staged = s3_stage_object_verified(
        creds,
        region,
        &artifacts_bucket,
        &worker_key,
        &manifest_dir.join(&manifest.worker_zip_file),
        &manifest.worker_zip_sha256_hex,
        opts.dry_run,
    )?;
    let layer_staged = s3_stage_object_verified(
        creds,
        region,
        &artifacts_bucket,
        &layer_key,
        &manifest_dir.join(&manifest.r882_layer_zip_file),
        &manifest.r882_layer_sha256_hex,
        opts.dry_run,
    )?;
    let all_artifacts_staged = bucket_exists && edge_staged && worker_staged && layer_staged;

    // 3. Determine existing stack state & capture previous :live alias versions for <1s rollback
    let existing_stack = read_stack_outputs_map(region, &stack_name)?;
    let is_fresh_or_review = matches!(
        existing_stack.as_ref().map(|s| s.status.as_str()),
        None | Some("REVIEW_IN_PROGRESS")
    );
    let previous_versions = if !is_fresh_or_review {
        read_current_live_aliases(region, &stack_name)
    } else {
        if opts.decommission_unstacked_prototype && !opts.dry_run {
            decommission_unstacked_prototype_resources(region, &stack_name)?;
        }
        None
    };

    let cs_type = match existing_stack.as_ref().map(|s| s.status.as_str()) {
        None | Some("REVIEW_IN_PROGRESS") => "CREATE",
        Some("ROLLBACK_COMPLETE") => {
            if !opts.dry_run {
                let _ = run_aws_json(
                    region,
                    &[
                        "cloudformation",
                        "delete-stack",
                        "--stack-name",
                        &stack_name,
                    ],
                )?;
                let del_deadline = Instant::now() + Duration::from_secs(60);
                while Instant::now() < del_deadline {
                    match read_stack_outputs_map(region, &stack_name)? {
                        None => break,
                        Some(snap) if snap.status == "DELETE_COMPLETE" => break,
                        Some(_) => thread::sleep(Duration::from_millis(500)),
                    }
                }
            }
            "CREATE"
        }
        Some(_) => "UPDATE",
    };

    // 4. Synthesize release-bound CloudFormation template inside a private 0700 stage dir
    let app_pubkeys_param = vault.app_auth_pubkeys_b64.clone().unwrap_or_default();
    let edge_cfg_tag = if !app_pubkeys_param.is_empty() || app_auth_pubkeys_override.is_some() {
        let mut h = Sha256::new();
        h.update(vault.edge_version_id.as_bytes());
        h.update(b":");
        h.update(app_pubkeys_param.as_bytes());
        Some(hex::encode(&h.finalize()[..4]))
    } else {
        None
    };
    let synthesized_json = synthesize_cell_template_for_release_with_edge_tag(
        &manifest.release_id,
        edge_cfg_tag.as_deref(),
    )?;
    struct SynthStageGuard(PathBuf);
    impl Drop for SynthStageGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let synth_dir = manifest_dir.join(format!(
        ".cfn-synth-{}-{}-{}",
        region,
        std::process::id(),
        now_epoch_ms()
    ));
    fs::create_dir_all(&synth_dir).map_err(|e| format!("create {}: {e}", synth_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&synth_dir, fs::Permissions::from_mode(0o700));
    }
    let _synth_guard = SynthStageGuard(synth_dir.clone());
    let synth_path = synth_dir.join(format!(
        "synthesized-{stack_name}-{region}-{}.json",
        manifest.release_id
    ));
    fs::write(&synth_path, &synthesized_json)
        .map_err(|e| format!("write {}: {e}", synth_path.display()))?;
    let template_arg = format!("file://{}", synth_path.display());

    // In `--dry-run`, if the stack does not yet exist (`CREATE` would leave/delete a REVIEW_IN_PROGRESS
    // stack) or artifacts/secrets are not yet created in AWS, validate the synthesized template strictly
    // read-only (`cloudformation validate-template`) with zero AWS resource mutations (`DEP-SEC-02`).
    if opts.dry_run && (cs_type == "CREATE" || !all_artifacts_staged || vault.secret_arn.is_empty())
    {
        let val_res = run_aws_json(
            region,
            &[
                "cloudformation",
                "validate-template",
                "--template-body",
                &template_arg,
            ],
        );
        let _ = fs::remove_file(&synth_path);
        val_res?;
        let tpl_val: serde_json::Value =
            serde_json::from_str(&synthesized_json).map_err(crate::error::err_to_string)?;
        let planned_resources: Vec<String> = tpl_val
            .get("Resources")
            .and_then(|r| r.as_object())
            .map(|m| {
                m.iter()
                    .map(|(lid, obj)| {
                        let rtype = obj.get("Type").and_then(|t| t.as_str()).unwrap_or("?");
                        format!("Plan {lid} ({rtype})")
                    })
                    .collect()
            })
            .unwrap_or_default();
        return Ok(CarbonRegionDeployReceipt {
            region: region.to_owned(),
            stack_name,
            environment: opts.environment.clone(),
            release_id: manifest.release_id.clone(),
            dry_run: true,
            change_set_action_count: planned_resources.len(),
            change_set_summary: planned_resources,
            artifacts_bucket,
            cas_bucket: None,
            edge_endpoint_url: None,
            public_edge_endpoint_url: None,
            engine_kek_pubkey_b64: Some(vault.engine_kek_pubkey_b64),
            live_versions: previous_versions.clone(),
            previous_versions,
            r882_layer_version_arn: None,
            redteam_passed_vectors: None,
            redteam_total_vectors: None,
            wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        });
    }

    let cas_expire_days = if opts.environment == "prod" { "7" } else { "1" };
    let log_retention_days = if opts.environment == "prod" {
        "30"
    } else {
        "14"
    };
    let cs_name = format!("cs-{}-{}", manifest.release_id, now_epoch_ms() / 1_000);

    let params = [
        format!(
            "ParameterKey=EnvironmentName,ParameterValue={}",
            opts.environment
        ),
        format!(
            "ParameterKey=ReleaseId,ParameterValue={}",
            manifest.release_id
        ),
        format!("ParameterKey=ArtifactsBucket,ParameterValue={artifacts_bucket}"),
        format!("ParameterKey=EdgeZipKey,ParameterValue={edge_key}"),
        format!(
            "ParameterKey=EdgeCodeSha256B64,ParameterValue={}",
            manifest.edge_code_sha256_b64
        ),
        format!("ParameterKey=WorkerZipKey,ParameterValue={worker_key}"),
        format!(
            "ParameterKey=WorkerCodeSha256B64,ParameterValue={}",
            manifest.worker_code_sha256_b64
        ),
        format!("ParameterKey=R882LayerZipKey,ParameterValue={layer_key}"),
        format!(
            "ParameterKey=R882BundleDigest,ParameterValue={}",
            manifest.r882_bundle_digest
        ),
        format!(
            "ParameterKey=SecretsArn,ParameterValue={}",
            vault.secret_arn
        ),
        format!(
            "ParameterKey=SecretsVersionId,ParameterValue={}",
            vault.version_id
        ),
        format!(
            "ParameterKey=EdgeSecretsArn,ParameterValue={}",
            vault.edge_secret_arn
        ),
        format!(
            "ParameterKey=EdgeSecretsVersionId,ParameterValue={}",
            vault.edge_version_id
        ),
        format!(
            "ParameterKey=EngineKekPubkeyB64,ParameterValue={}",
            vault.engine_kek_pubkey_b64
        ),
        format!("ParameterKey=AppAuthPubkeysB64,ParameterValue={app_pubkeys_param}"),
        format!("ParameterKey=CasExpireDays,ParameterValue={cas_expire_days}"),
        format!("ParameterKey=LogRetentionDays,ParameterValue={log_retention_days}"),
    ];

    let mut cs_args: Vec<&str> = vec![
        "cloudformation",
        "create-change-set",
        "--stack-name",
        &stack_name,
        "--change-set-name",
        &cs_name,
        "--change-set-type",
        cs_type,
        "--template-body",
        &template_arg,
        "--capabilities",
        "CAPABILITY_NAMED_IAM",
        "--parameters",
    ];
    for p in &params {
        cs_args.push(p.as_str());
    }

    let create_res = run_aws_json(region, &cs_args);
    let _ = fs::remove_file(&synth_path);
    create_res?;

    let (has_changes, summary) = match wait_for_change_set(region, &stack_name, &cs_name) {
        Ok(res) => res,
        Err(err) => {
            let _ = run_aws_allow_missing(
                region,
                &[
                    "cloudformation",
                    "delete-change-set",
                    "--stack-name",
                    &stack_name,
                    "--change-set-name",
                    &cs_name,
                ],
            );
            if cs_type == "CREATE" && is_fresh_or_review {
                let _ = run_aws_allow_missing(
                    region,
                    &[
                        "cloudformation",
                        "delete-stack",
                        "--stack-name",
                        &stack_name,
                    ],
                );
            }
            return Err(err);
        }
    };

    if opts.dry_run {
        let _ = run_aws_allow_missing(
            region,
            &[
                "cloudformation",
                "delete-change-set",
                "--stack-name",
                &stack_name,
                "--change-set-name",
                &cs_name,
            ],
        )?;
        if cs_type == "CREATE" && is_fresh_or_review {
            let _ = run_aws_allow_missing(
                region,
                &[
                    "cloudformation",
                    "delete-stack",
                    "--stack-name",
                    &stack_name,
                ],
            )?;
        }
        return Ok(CarbonRegionDeployReceipt {
            region: region.to_owned(),
            stack_name,
            environment: opts.environment.clone(),
            release_id: manifest.release_id.clone(),
            dry_run: true,
            change_set_action_count: summary.len(),
            change_set_summary: summary,
            artifacts_bucket,
            cas_bucket: None,
            edge_endpoint_url: None,
            public_edge_endpoint_url: None,
            engine_kek_pubkey_b64: Some(vault.engine_kek_pubkey_b64),
            live_versions: previous_versions.clone(),
            previous_versions,
            r882_layer_version_arn: None,
            redteam_passed_vectors: None,
            redteam_total_vectors: None,
            wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        });
    }

    if has_changes {
        let _ = run_aws_json(
            region,
            &[
                "cloudformation",
                "execute-change-set",
                "--stack-name",
                &stack_name,
                "--change-set-name",
                &cs_name,
            ],
        )?;
        wait_for_stack_completion(region, &stack_name)?;
    } else {
        let _ = run_aws_allow_missing(
            region,
            &[
                "cloudformation",
                "delete-change-set",
                "--stack-name",
                &stack_name,
                "--change-set-name",
                &cs_name,
            ],
        )?;
    }

    let snap = read_stack_outputs_map(region, &stack_name)?
        .ok_or_else(|| format!("stack '{stack_name}' missing after execution"))?;
    let outputs = snap.outputs;

    let edge_url = outputs
        .get("EdgeEndpointUrl")
        .cloned()
        .ok_or_else(|| format!("stack '{stack_name}' missing EdgeEndpointUrl"))?;
    let public_edge_url = outputs.get("PublicEdgeEndpointUrl").cloned();
    let cas_bucket = outputs.get("CasBucketName").cloned();
    let layer_arn = outputs.get("R882LayerVersionArn").cloned();
    let live_versions = Some(CarbonLiveAliasVersions {
        edge_version: outputs
            .get("EdgePublishedVersion")
            .cloned()
            .unwrap_or_default(),
        engine_version: outputs
            .get("EnginePublishedVersion")
            .cloned()
            .unwrap_or_default(),
        eval_version: outputs
            .get("EvalPublishedVersion")
            .cloned()
            .unwrap_or_default(),
    });

    // Reconcile `:live` aliases when a previous emergency rollback mutated Lambda aliases
    // out-of-band without changing CloudFormation's stack model (`!has_changes`).
    if !has_changes {
        if let Some(ref expected_live) = live_versions {
            if !expected_live.edge_version.is_empty()
                && !expected_live.engine_version.is_empty()
                && !expected_live.eval_version.is_empty()
            {
                let actual_live = read_current_live_aliases(region, &stack_name);
                let drifted = actual_live.as_ref().is_some_and(|cur| {
                    cur.edge_version != expected_live.edge_version
                        || cur.engine_version != expected_live.engine_version
                        || cur.eval_version != expected_live.eval_version
                });
                if drifted {
                    let _ = execute_carbon_rollback(&CarbonRollbackOptions {
                        environment: opts.environment.clone(),
                        region: region.to_owned(),
                        target_versions: Some(expected_live.clone()),
                    })?;
                }
            }
        }
    }

    // 5. Mandatory post-deploy live verification (/ping + 16-vector red-team suite on BOTH IAM & Public doors) + auto-rollback!
    let verify_outcome: Result<(Option<usize>, Option<usize>), String> = (|| {
        let mut gate_urls = vec![("iam_fn_url", edge_url.as_str())];
        if let Some(pub_u) = public_edge_url.as_deref() {
            gate_urls.push(("public_http_api", pub_u));
        }
        let mut passed_sum = 0usize;
        let mut total_sum = 0usize;
        for (door_label, url) in &gate_urls {
            let (probed_client, probed) =
                CarbonThinClient::connect_fastest(vec![(*url).to_owned()])?;
            let first = probed.first().ok_or_else(|| {
                format!("post-deploy /ping ({door_label}) returned empty probe for {url}")
            })?;
            if first.engine_kek_pubkey_b64.as_deref() != Some(vault.engine_kek_pubkey_b64.as_str())
            {
                return Err(format!(
                    "post-deploy /ping ({door_label}) engine_kek_pubkey_b64 mismatch against vault"
                ));
            }
            if let Some(ver) = probed_client.fetch_server_version()? {
                if !ver.git_sha.is_empty()
                    && ver.git_sha != "0000000000000000000000000000000000000000"
                    && ver.git_sha != manifest.git_sha
                {
                    return Err(format!(
                        "post-deploy /v1/carbon/version ({door_label}) git_sha mismatch: server={}, manifest={}",
                        ver.git_sha, manifest.git_sha
                    ));
                }
                if let Some(rel_id) = ver.release_id.as_deref() {
                    if rel_id != manifest.release_id {
                        return Err(format!(
                            "post-deploy /v1/carbon/version ({door_label}) release_id mismatch: server={rel_id}, manifest={}",
                            manifest.release_id
                        ));
                    }
                }
            }
            if !opts.skip_redteam {
                let report = match execute_carbon_redteam_suite(url) {
                    Ok(r) => r,
                    Err(err) if err.contains("anonymous_disabled") => {
                        let u_a = ulid::Ulid::new().to_string().to_ascii_lowercase();
                        let u_b = ulid::Ulid::new().to_string().to_ascii_lowercase();
                        let tok_a = mint_capability_token_from_vault(
                            &opts.environment,
                            region,
                            &format!("rt_a_{}", &u_a[u_a.len().saturating_sub(16)..]),
                            jope_api::carbon::CarbonTierWire::Free,
                            900,
                            false,
                        )?;
                        let tok_b = mint_capability_token_from_vault(
                            &opts.environment,
                            region,
                            &format!("rt_b_{}", &u_b[u_b.len().saturating_sub(16)..]),
                            jope_api::carbon::CarbonTierWire::Free,
                            900,
                            false,
                        )?;
                        execute_carbon_redteam_suite_with_tokens(url, Some((tok_a, tok_b)))?
                    }
                    Err(err) => return Err(err),
                };
                if !report.all_passed {
                    let failed_ids: Vec<String> = report
                        .vectors
                        .iter()
                        .filter(|v| !v.passed)
                        .map(|v| format!("{} ({})", v.vector_id, v.evidence))
                        .collect();
                    return Err(format!(
                        "post-deploy live redteam on {door_label} failed ({}/{} passed): {}",
                        report.passed_vectors,
                        report.total_vectors,
                        failed_ids.join("; ")
                    ));
                }
                passed_sum = report.passed_vectors;
                total_sum = report.total_vectors;
            }
        }
        if opts.skip_redteam {
            return Ok((None, None));
        }
        Ok((Some(passed_sum), Some(total_sum)))
    })();

    let (rt_passed, rt_total) = match verify_outcome {
        Ok(counts) => counts,
        Err(gate_err) => {
            if let Some(prev) = &previous_versions {
                let _ = execute_carbon_rollback(&CarbonRollbackOptions {
                    environment: opts.environment.clone(),
                    region: region.to_owned(),
                    target_versions: Some(prev.clone()),
                });
                return Err(format!(
                    "{gate_err} — automatically rolled :live aliases back to edge={}, engine={}, eval={}",
                    prev.edge_version, prev.engine_version, prev.eval_version
                ));
            }
            return Err(gate_err);
        }
    };

    Ok(CarbonRegionDeployReceipt {
        region: region.to_owned(),
        stack_name,
        environment: opts.environment.clone(),
        release_id: manifest.release_id.clone(),
        dry_run: false,
        change_set_action_count: summary.len(),
        change_set_summary: summary,
        artifacts_bucket,
        cas_bucket,
        edge_endpoint_url: Some(edge_url),
        public_edge_endpoint_url: public_edge_url,
        engine_kek_pubkey_b64: Some(vault.engine_kek_pubkey_b64),
        live_versions,
        previous_versions,
        r882_layer_version_arn: layer_arn,
        redteam_passed_vectors: rt_passed,
        redteam_total_vectors: rt_total,
        wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
    })
}

/// Execute single-region or multi-region wave deployment from a verified `release-manifest.json`.
pub fn execute_carbon_deploy(opts: &CarbonDeployOptions) -> Result<CarbonDeployReport, String> {
    execute_carbon_deploy_with_app_pubkeys(opts, None)
}

/// Execute single-region or multi-region wave deployment with an optional `app_auth_pubkeys_b64`
/// override (`[active, previous]` comma-separated Base64 uncompressed P-256 public keys, or `CLEAR`).
pub fn execute_carbon_deploy_with_app_pubkeys(
    opts: &CarbonDeployOptions,
    app_auth_pubkeys_b64: Option<&str>,
) -> Result<CarbonDeployReport, String> {
    let _ = normalize_app_pubkeys_override(app_auth_pubkeys_b64)?;
    if !matches!(opts.environment.as_str(), "dev" | "staging" | "prod") {
        return Err(format!(
            "unsupported environment '{}' (expected dev, staging, or prod)",
            opts.environment
        ));
    }
    let manifest_bytes = fs::read(&opts.manifest_path)
        .map_err(|e| format!("read {}: {e}", opts.manifest_path.display()))?;
    let manifest: CarbonReleaseManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("parse {}: {e}", opts.manifest_path.display()))?;
    let manifest_dir = opts
        .manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let creds = CarbonAwsCredentials::from_env()?;
    let target_regions = expand_region_targets(&opts.regions)?;
    let first_region = target_regions
        .first()
        .cloned()
        .unwrap_or_else(|| "us-east-2".to_owned());
    let caller = run_aws_json(&first_region, &["sts", "get-caller-identity"])?;
    let account_id = caller
        .get("Account")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "sts get-caller-identity missing Account".to_owned())?
        .to_owned();

    // Resolve canonical global `auth_secret_hex` from primary vault region (`us-east-2`) so every
    // regional cell verifies the same `ctk1.*` capability tokens while retaining isolated regional KEKs.
    let primary_vault = ensure_cell_secrets_in_vault_with_pubkeys(
        &creds,
        "us-east-2",
        &opts.environment,
        opts.bootstrap_secrets,
        None,
        app_auth_pubkeys_b64,
        opts.dry_run,
    )?;

    let deploy_ctx = RegionDeployContext {
        creds: &creds,
        account_id: &account_id,
        manifest: &manifest,
        manifest_dir: &manifest_dir,
        opts,
        global_auth_secret_hex: &primary_vault.auth_secret_hex,
        app_auth_pubkeys_override: app_auth_pubkeys_b64,
    };
    let mut receipts = Vec::with_capacity(target_regions.len());
    if let Some(canary_region) = target_regions.first() {
        let canary_receipt = deploy_single_region(&deploy_ctx, canary_region)?;
        // Progressive wave promotion gate: before rolling out beyond the first (canary) region,
        // verify a live zero-knowledge `r882` GP inference & cryptographic shred round-trip!
        if target_regions.len() > 1 && !opts.dry_run && !opts.skip_redteam {
            if let Some(canary_url) = canary_receipt
                .public_edge_endpoint_url
                .as_ref()
                .or(canary_receipt.edge_endpoint_url.as_ref())
            {
                let canary_auth_tok = CarbonAuthVerifier::from_hex(&primary_vault.auth_secret_hex)
                    .map_err(crate::error::err_to_string)
                    .and_then(|v| {
                        v.mint_token(&CarbonCapabilityClaims {
                            tenant_id: "canary_wave_gate".to_owned(),
                            tier: jope_api::carbon::CarbonTierWire::Free,
                            exp_epoch_secs: (now_epoch_ms() / 1_000).saturating_add(300),
                            admin: false,
                        })
                        .map_err(crate::error::err_to_string)
                    })?;
                let canary_gate = execute_carbon_cli_run(&CarbonCliRunOptions {
                    endpoints: vec![canary_url.clone()],
                    csv_path: None,
                    target_column: "y".to_owned(),
                    synthetic_rows: 48,
                    synthetic_features: 2,
                    tenant_id: "canary_wave_gate".to_owned(),
                    tier: jope_api::carbon::CarbonTierWire::Free,
                    effort: jope_api::Effort::Fast,
                    objective: "rmse".to_owned(),
                    seed: 42,
                    holdout_fraction: 0.20,
                    stop_after_ms: None,
                    resume_after_stop: false,
                    predict_rows: 4,
                    auth_token: Some(canary_auth_tok),
                    shred_after_run: true,
                })
                .map_err(|e| {
                    format!(
                        "canary region '{canary_region}' failed post-deploy E2E inference gate before multi-region wave promotion: {e}"
                    )
                })
                .and_then(|e2e_receipt| {
                    if !e2e_receipt.encrypted_cuen || !e2e_receipt.shredded_after_run {
                        Err(format!(
                            "canary region '{canary_region}' E2E gate violated zero-knowledge encryption/shred invariant"
                        ))
                    } else {
                        Ok(())
                    }
                });
                if let Err(canary_err) = canary_gate {
                    if let Some(prev) = &canary_receipt.previous_versions {
                        let _ = execute_carbon_rollback(&CarbonRollbackOptions {
                            environment: opts.environment.clone(),
                            region: canary_region.clone(),
                            target_versions: Some(prev.clone()),
                        });
                        return Err(format!(
                            "{canary_err} — automatically rolled :live aliases back to edge={}, engine={}, eval={}",
                            prev.edge_version, prev.engine_version, prev.eval_version
                        ));
                    }
                    return Err(canary_err);
                }
            }
        }
        receipts.push(canary_receipt);

        let ctx_ref = &deploy_ctx;
        for chunk in target_regions[1..].chunks(4) {
            let chunk_results: Vec<Result<CarbonRegionDeployReceipt, String>> =
                thread::scope(|s| {
                    let mut handles = Vec::with_capacity(chunk.len());
                    for region in chunk {
                        let reg = region.as_str();
                        handles.push(s.spawn(move || deploy_single_region(ctx_ref, reg)));
                    }
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join().unwrap_or_else(|_| {
                                Err("regional deploy worker panicked".to_owned())
                            })
                        })
                        .collect()
                });
            for res in chunk_results {
                receipts.push(res?);
            }
        }
    }

    Ok(CarbonDeployReport {
        timestamp_epoch_ms: now_epoch_ms(),
        environment: opts.environment.clone(),
        release_id: manifest.release_id,
        git_sha: manifest.git_sha,
        dry_run: opts.dry_run,
        regions: receipts,
    })
}
