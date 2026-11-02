use super::*;

mod derivation;
mod systemd;

pub(super) use derivation::{
    derive_rollback_baseline, verify_rollback_candidate, verify_rollback_capsule,
    verify_rollback_derivation,
};
pub(super) use systemd::{
    git_blob, parse_systemd_artifact, rollback_bundle_files, rollback_derivation_path,
    systemd_closure, transform_rollback_unit, unchanged_inventory_root, verify_intrinsic_existing,
};

pub(super) fn verify_existing(root: &Path, manifest: &Value) -> Result<()> {
    ensure!(manifest["runtime"] == "rust", "bundle runtime is not Rust");
    ensure!(
        manifest["operational_python_allowed"] == false,
        "bundle permits operational Python"
    );
    ensure!(
        manifest
            .get("clean_detached_source")
            .and_then(Value::as_bool)
            == Some(true)
            && manifest
                .get("integration_commit")
                .and_then(Value::as_str)
                .is_some_and(|commit| {
                    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
                }),
        "bundle lacks clean detached integration provenance"
    );
    ensure!(
        manifest
            .pointer("/prime_support/source_bundle_sha256")
            .and_then(Value::as_str)
            == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "bundle does not bind immutable Prime support"
    );
    let required_nodes = manifest
        .get("required_sync_nodes")
        .and_then(Value::as_array)
        .context("bundle lacks five-node synchronization contract")?;
    ensure!(
        required_nodes
            == &GPU_SYNC_NODES
                .iter()
                .map(|node| Value::String((*node).to_owned()))
                .collect::<Vec<_>>(),
        "bundle synchronization node set drift"
    );
    ensure!(
        manifest
            .pointer("/five_node_synchronization/schema")
            .and_then(Value::as_str)
            == Some("JopeBundleSynchronizationContract.v1")
            && manifest
                .pointer("/five_node_synchronization/receipt_schema")
                .and_then(Value::as_str)
                == Some("JopeBundleSynchronizationReceipt.v1")
            && manifest
                .pointer("/five_node_synchronization/activation_requires_complete")
                .and_then(Value::as_bool)
                == Some(true)
            && manifest
                .pointer("/five_node_synchronization/required_nodes")
                .and_then(Value::as_array)
                == Some(required_nodes)
            && manifest
                .pointer("/five_node_synchronization/receipt_relative_path")
                .and_then(Value::as_str)
                == manifest
                    .get("bundle_sha256")
                    .and_then(Value::as_str)
                    .map(|identity| format!("synchronization/{identity}.v1.json"))
                    .as_deref(),
        "bundle synchronization receipt contract drift"
    );
    let rows = manifest["files"].as_array().context("manifest files")?;
    let service_manifest: ServiceManifest = serde_json::from_value(
        manifest
            .get("service_manifest")
            .cloned()
            .context("bundle lacks embedded service manifest")?,
    )?;
    validate_recovery_service_manifest(&service_manifest)?;
    let identity_rows: Vec<_> = rows
        .iter()
        .map(|row| {
            json!({
                "target": row["target"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let computed_identity = sha256_bytes(&serde_json::to_vec(&identity_rows)?);
    ensure!(
        manifest["bundle_sha256"].as_str() == Some(computed_identity.as_str()),
        "bundle manifest identity drift"
    );
    ensure!(
        root.file_name().and_then(|name| name.to_str()) == Some(computed_identity.as_str()),
        "bundle directory does not match its identity"
    );
    for row in rows {
        let target = row["target"].as_str().context("file target")?;
        let expected = row["sha256"].as_str().context("file sha256")?;
        ensure!(
            sha256_file(&root.join(target))? == expected,
            "existing bundle content drift: {target}"
        );
        if target.ends_with(".service") {
            validate_systemd_unit(&fs::read_to_string(root.join(target))?)?;
        }
        if target.ends_with(".py") {
            let role = row
                .get("python_role")
                .and_then(Value::as_str)
                .context("unclassified Python in bundle")?;
            validate_manifest_python_target(target, role)?;
        }
        if target.ends_with(".sh") {
            let contents = fs::read_to_string(root.join(target))?;
            if target == "pytorch/encoding/runner.sh" {
                validate_encoding_runner(&contents)?;
            } else {
                validate_no_python_runtime(&contents)?;
            }
        }
    }
    for service in service_manifest.services.iter().filter(|row| row.active) {
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/bin/{}", service.binary).as_str())
            }) && rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/systemd/{}", service.unit).as_str())
            }),
            "bundle service/unit closure is incomplete for {}",
            service.unit
        );
    }
    let prime_files = manifest
        .pointer("/prime_support/files")
        .and_then(Value::as_array)
        .context("bundle lacks Prime per-file provenance")?;
    let mut canonical_prime_files =
        serde_json::from_value::<Vec<PrimeSupportFile>>(Value::Array(prime_files.clone()))?;
    canonical_prime_files.sort_by(|left, right| {
        left.source_bundle_target
            .cmp(&right.source_bundle_target)
            .then_with(|| left.bundle_target.cmp(&right.bundle_target))
    });
    let computed_prime_tree_sha256 = prime_support_identity(&canonical_prime_files)?;
    ensure!(
        manifest
            .pointer("/prime_support/package_tree_sha256")
            .and_then(Value::as_str)
            == Some(computed_prime_tree_sha256.as_str()),
        "Prime package-tree digest drift"
    );
    for prime in prime_files {
        let target = prime
            .get("bundle_target")
            .and_then(Value::as_str)
            .context("Prime support row lacks bundle target")?;
        let digest = prime
            .get("sha256")
            .and_then(Value::as_str)
            .context("Prime support row lacks digest")?;
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str) == Some(target)
                    && row.get("sha256").and_then(Value::as_str) == Some(digest)
            }),
            "Prime support row is absent from bundle inventory: {target}"
        );
    }
    for required in [
        "pytorch/jope/__init__.py",
        "pytorch/jope/policy/model_v3.py",
        "pytorch/jope/policy/decode.py",
        "pytorch/proposals/test_exact_context_proposals.py",
    ] {
        ensure!(
            root.join(required).is_file(),
            "bundle lacks required JOPE proposal dependency: {required}"
        );
    }
    Ok(())
}

pub(super) fn validate_manifest_python_target(target: &str, role: &str) -> Result<()> {
    if role == "pytorch_dependency_package"
        && target.starts_with("pytorch/jope/")
        && target.ends_with(".py")
    {
        return Ok(());
    }
    ensure!(
        PYTORCH_ALLOWLIST
            .iter()
            .any(|(_, allowed_target, allowed_role)| {
                target == *allowed_target && role == *allowed_role
            }),
        "unallowlisted Python in bundle: {target}"
    );
    Ok(())
}

pub(super) fn validate_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    ensure!(
        manifest.schema == "JopeSelfplayServiceManifest.v3",
        "unsupported service manifest schema"
    );
    let binaries = manifest.binaries.iter().collect::<BTreeSet<_>>();
    ensure!(
        binaries.len() == manifest.binaries.len() && !binaries.is_empty(),
        "service manifest binary inventory is empty or duplicated"
    );
    let mut units = BTreeSet::new();
    for row in &manifest.services {
        ensure!(
            row.unit.ends_with(".service")
                && !row.unit.contains('/')
                && !row.owner.trim().is_empty(),
            "service manifest contains an invalid unit or owner"
        );
        ensure!(
            units.insert(row.unit.as_str()),
            "service manifest repeats unit {}",
            row.unit
        );
        ensure!(
            binaries.contains(&row.binary),
            "service {} references unbundled binary {}",
            row.unit,
            row.binary
        );
    }
    Ok(())
}

pub(super) fn validate_recovery_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    validate_service_manifest(manifest)?;
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-reporting.service" && row.binary == "jope_reporting" && row.active
        }),
        "recovery bundle must install and activate jope-reporting.service"
    );
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-selfplay-expansion.service"
                && row.binary == "jope_selfplay_expansion_driver"
                && !row.active
        }),
        "recovery bundle must carry expansion as an inactive Encoder V2-gated unit"
    );
    Ok(())
}

pub(super) fn ensure_bundle_target(root: &Path, target: &Path) -> Result<()> {
    ensure!(
        target.starts_with(root),
        "bundle target escaped temporary directory"
    );
    Ok(())
}
