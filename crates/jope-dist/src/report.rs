//! Read-side commands, independent of the executable inside the package (SPECIFICATION.md
//! §10.4, §12.4, §16.4): `inspect`, `verify`, `size-report`, `sbom-models` and `provenance`.
use anyhow::{ensure, Context, Result};
use jope_assets::verify::{self, TrustedKey};
use jope_assets::Package;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Keys from `--trusted-key PEM` files, the development key with `--development-key`, and the
/// compiled-in release key unless `--no-release-key`.
pub fn trusted(pems: &[PathBuf], development: bool, release: bool) -> Result<Vec<TrustedKey>> {
    let mut out = if release {
        verify::release_keys()
    } else {
        Vec::new()
    };
    for path in pems {
        let pem =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        out.push(
            TrustedKey::from_pem(&pem, "release")
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
        );
    }
    if development {
        out.push(verify::development_key());
    }
    ensure!(!out.is_empty(), "no trusted key");
    Ok(out)
}

fn open(file: &Path, trusted: &[TrustedKey]) -> Result<Package> {
    Package::open_with_keys(file, trusted)
        .with_context(|| format!("open {}", file.display()))?
        .with_context(|| format!("{} carries no package", file.display()))
}

/// The manifest, footer and signer, without decoding any asset.
pub fn inspect(file: &Path, trusted: &[TrustedKey]) -> Result<Value> {
    let package = open(file, trusted)?;
    let footer = package.footer();
    Ok(json!({
        "schema": "jope.appliance-inspect/v1",
        "file": file.display().to_string(),
        "signer": {"id": package.signer().id.to_string(), "role": package.signer().role},
        "footer": {
            "base_executable_len": footer.base_executable_len,
            "payload_offset": footer.payload_offset(),
            "manifest_offset": footer.manifest_offset,
            "manifest_len": footer.manifest_len,
            "signed_prefix_len": footer.signed_prefix_len(),
            "manifest_sha256": hex::encode(footer.manifest_sha256),
        },
        "manifest": package.manifest(),
    }))
}

/// Every byte: the executable hash, padding and gaps, every asset, the signature.
pub fn verify_file(file: &Path, trusted: &[TrustedKey]) -> Result<Value> {
    let package = open(file, trusted)?;
    let report = package.verify_full()?;
    Ok(json!({
        "schema": "jope.appliance-verify/v1",
        "ok": true,
        "file": file.display().to_string(),
        "profile": package.manifest().profile,
        "scope": package.manifest().scope.as_str(),
        "signer": {"id": package.signer().id.to_string(), "role": package.signer().role},
        "report": report,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Budgets {
    schema: String,
    tolerance_percent: u64,
    #[serde(rename = "profile")]
    profiles: BTreeMap<String, ProfileBudget>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileBudget {
    base_executable_bytes: u64,
    final_executable_bytes: u64,
    #[serde(default)]
    note: String,
}

/// Sizes of every region and asset; with a budget file, fail above budget + tolerance.
pub fn size_report(
    file: &Path,
    trusted: &[TrustedKey],
    budget: Option<&Path>,
) -> Result<(Value, bool)> {
    let package = open(file, trusted)?;
    let manifest = package.manifest();
    let footer = package.footer();
    let entries: Vec<Value> = manifest
        .entries
        .iter()
        .map(|e| {
            json!({
                "id": e.id,
                "decoded_bytes": e.bytes,
                "stored_bytes": e.stored_bytes,
                "ratio": if e.bytes == 0 { 1.0 } else { e.stored_bytes as f64 / e.bytes as f64 },
                "codec": e.codec,
            })
        })
        .collect();
    let decoded: u64 = manifest.entries.iter().map(|e| e.bytes).sum();
    let stored: u64 = manifest.entries.iter().map(|e| e.stored_bytes).sum();
    let mut ok = true;
    let mut checks = Vec::new();
    if let Some(path) = budget {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let budgets: Budgets =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            budgets.schema == "jope.appliance-size-budgets/v1",
            "{} has the wrong schema",
            path.display()
        );
        let limits = budgets.profiles.get(&manifest.profile).with_context(|| {
            format!(
                "{} has no budget for profile {}",
                path.display(),
                manifest.profile
            )
        })?;
        let factor = 1.0 + budgets.tolerance_percent as f64 / 100.0;
        for (what, actual, limit) in [
            (
                "base_executable_bytes",
                footer.base_executable_len,
                limits.base_executable_bytes,
            ),
            (
                "final_executable_bytes",
                package.file_len(),
                limits.final_executable_bytes,
            ),
        ] {
            let pass = actual as f64 <= limit as f64 * factor;
            ok &= pass;
            checks.push(json!({"what": what, "bytes": actual, "budget": limit, "tolerance_percent": budgets.tolerance_percent, "pass": pass, "note": limits.note}));
        }
    }
    Ok((
        json!({
            "schema": "jope.appliance-size/v1",
            "profile": manifest.profile,
            "base_executable_bytes": footer.base_executable_len,
            "raw_model_bytes": manifest.entries.iter().filter(|e| e.role == "model").map(|e| e.bytes).sum::<u64>(),
            "stored_model_bytes": manifest.entries.iter().filter(|e| e.role == "model").map(|e| e.stored_bytes).sum::<u64>(),
            "decoded_asset_bytes": decoded,
            "stored_asset_bytes": stored,
            "payload_bytes": footer.manifest_offset - footer.payload_offset(),
            "manifest_bytes": footer.manifest_len,
            "signature_footer_bytes": jope_assets::SIGNATURE_LEN + jope_assets::FOOTER_LEN,
            "final_executable_bytes": package.file_len(),
            "gpu_library_bytes": 0,
            "entries": entries,
            "budget_checks": checks,
            "ok": ok,
        }),
        ok,
    ))
}

/// CycloneDX 1.6 components for every packaged asset.
pub fn sbom_models(file: &Path, trusted: &[TrustedKey]) -> Result<Value> {
    let package = open(file, trusted)?;
    let manifest = package.manifest();
    let components: Vec<Value> = manifest
        .entries
        .iter()
        .map(|e| {
            json!({
                "type": if e.role == "model" { "machine-learning-model" } else { "data" },
                "bom-ref": format!("jopedime:{}", e.id),
                "name": e.id,
                "hashes": [{"alg": "SHA-256", "content": e.sha256}],
                "licenses": [{"expression": e.license.spdx}],
                "properties": [
                    {"name": "jope:origin", "value": e.origin},
                    {"name": "jope:role", "value": e.role},
                    {"name": "jope:redistribution", "value": e.license.redistribution},
                    {"name": "jope:backends", "value": e.backends.join(",")},
                    {"name": "jope:codec", "value": e.codec},
                ],
            })
        })
        .collect();
    Ok(json!({
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": manifest.product,
                "version": manifest.product_version,
                "hashes": [{"alg": "SHA-256", "content": manifest.code.sha256}],
                "properties": [
                    {"name": "jope:profile", "value": manifest.profile},
                    {"name": "jope:scope", "value": manifest.scope.as_str()},
                    {"name": "jope:source_sha40", "value": manifest.source.sha40},
                ],
            },
        },
        "components": components,
    }))
}

pub struct ProvenanceInputs<'a> {
    pub tag: &'a str,
    pub builder_image: &'a str,
    pub builder_id: &'a str,
    pub cargo_profile: &'a str,
    pub features: &'a str,
}

/// The SLSA v1 provenance predicate for a sealed package, bound to its whole-file sha256. It is an
/// evidence asset beside the executable, never embedded in it (SPECIFICATION.md §10.3).
pub fn provenance(
    file: &Path,
    trusted: &[TrustedKey],
    inputs: &ProvenanceInputs<'_>,
) -> Result<Value> {
    let package = open(file, trusted)?;
    let report = package.verify_full()?;
    let m = package.manifest();
    let mut deps = vec![
        json!({"uri": format!("git+https://github.com/neverhuman/JopeDime@{}", m.source.sha40), "digest": {"gitCommit": m.source.sha40}}),
        json!({"uri": "Cargo.lock", "digest": {"sha256": m.source.cargo_lock_sha256}}),
        json!({"uri": format!("deploy/appliance/profiles/{}.toml", m.profile), "digest": {"sha256": m.source.profile_sha256}}),
        json!({"uri": "deploy/appliance/models.lock.json", "digest": {"sha256": m.source.models_lock_sha256}}),
    ];
    for e in &m.entries {
        deps.push(
            json!({"uri": format!("jope-asset://{}", e.origin), "digest": {"sha256": e.sha256}}),
        );
    }
    Ok(json!({
        "subject": [{"name": file.file_name().map(|n| n.to_string_lossy().into_owned()), "digest": {"sha256": report.file_sha256}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/neverhuman/JopeDime/deploy/appliance/build@v1",
                "externalParameters": {"tag": inputs.tag, "profile": m.profile, "scope": m.scope.as_str(), "source": m.source.sha40},
                "internalParameters": {
                    "builderImage": inputs.builder_image,
                    "cargoProfile": inputs.cargo_profile,
                    "features": inputs.features,
                    "target": m.target,
                    "baseExecutableSha256": m.code.sha256,
                },
                "resolvedDependencies": deps,
            },
            "runDetails": {
                "builder": {"id": inputs.builder_id},
                "metadata": {"invocationId": format!("{}-{}", inputs.tag, m.profile)},
            },
        },
    }))
}
