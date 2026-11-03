use super::*;

pub(crate) fn build(args: &Args) -> Result<Value> {
    let integration_commit = clean_detached_commit(&args.source)?;
    let service_manifest_path = args
        .service_manifest
        .clone()
        .unwrap_or_else(|| args.source.join(DEFAULT_SERVICE_MANIFEST));
    let service_manifest: ServiceManifest = serde_json::from_reader(std::io::BufReader::new(
        File::open(&service_manifest_path).with_context(|| {
            format!(
                "open declarative service manifest {}",
                service_manifest_path.display()
            )
        })?,
    ))?;
    validate_recovery_service_manifest(&service_manifest)?;
    let candidates = sources(args, &service_manifest, &service_manifest_path)?;
    let inventory = inventory(&candidates)?;
    let prime_support = prime_support_provenance(args, &candidates, &inventory)?;
    let identity = inventory_identity(&inventory)?;
    let final_dir = args.output_root.join(&identity);
    let manifest_path = final_dir.join("bundle.json");
    if manifest_path.is_file() {
        let _ = fs::remove_dir_all(rendered_units_dir(args));
        let manifest: Value =
            serde_json::from_reader(std::io::BufReader::new(File::open(&manifest_path)?))?;
        verify_existing(&final_dir, &manifest)?;
        return Ok(manifest);
    }

    fs::create_dir_all(&args.output_root)?;
    let temporary = args
        .output_root
        .join(format!(".{identity}.tmp-{}", std::process::id()));
    if temporary.exists() {
        bail!("temporary bundle already exists: {}", temporary.display());
    }
    fs::create_dir(&temporary)?;
    let result = (|| -> Result<Value> {
        for (candidate, row) in candidates.iter().zip(&inventory) {
            let destination = temporary.join(&row.target);
            ensure_bundle_target(&temporary, &destination)?;
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&candidate.source, &destination).with_context(|| {
                format!(
                    "copy {} to {}",
                    candidate.source.display(),
                    destination.display()
                )
            })?;
            ensure!(
                sha256_file(&destination)? == row.sha256,
                "bundle copy hash mismatch: {}",
                row.target
            );
        }
        let python_targets: Vec<_> = inventory
            .iter()
            .filter_map(|row| {
                row.python_role
                    .map(|role| json!({"target": row.target, "role": role}))
            })
            .collect();
        let active_units = service_manifest
            .services
            .iter()
            .filter(|row| row.active)
            .map(|row| row.unit.as_str())
            .collect::<Vec<_>>();
        let manifest = json!({
            "schema": "RuntimeBundle.v3",
            "bundle_sha256": identity,
            "runtime": "rust",
            "integration_commit": integration_commit,
            "clean_detached_source": true,
            "operational_python_allowed": false,
            "pytorch_python_allowlist": python_targets,
            "prime_support": prime_support,
            "required_sync_nodes": GPU_SYNC_NODES,
            "five_node_synchronization": {
                "schema": "JopeBundleSynchronizationContract.v1",
                "receipt_schema": "JopeBundleSynchronizationReceipt.v1",
                "receipt_relative_path": format!("synchronization/{identity}.v1.json"),
                "required_nodes": GPU_SYNC_NODES,
                "activation_requires_complete": true,
            },
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-exact-reused",
            "active_units": active_units,
            "service_manifest": service_manifest,
            "service_manifest_sha256": sha256_file(&service_manifest_path)?,
            "files": inventory,
        });
        atomic_json(&temporary.join("bundle.json"), &manifest)?;
        fs::rename(&temporary, &final_dir)?;
        verify_existing(&final_dir, &manifest)?;
        Ok(manifest)
    })();
    let _ = fs::remove_dir_all(rendered_units_dir(args));
    if result.is_err() && temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    result
}

pub(crate) fn git_output(source: &Path, arguments: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(source)
        .args(arguments)
        .output()
        .with_context(|| format!("run git in {}", source.display()))?;
    ensure!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub(crate) fn clean_detached_commit(source: &Path) -> Result<String> {
    ensure!(
        git_output(source, &["status", "--porcelain"])?.is_empty(),
        "runtime bundle source worktree is dirty"
    );
    ensure!(
        git_output(source, &["rev-parse", "--abbrev-ref", "HEAD"])? == "HEAD",
        "runtime bundle source must be a detached reviewed commit"
    );
    let commit = git_output(source, &["rev-parse", "HEAD"])?;
    ensure!(
        commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "runtime bundle integration commit is malformed"
    );
    Ok(commit)
}

#[path = "build/inputs.rs"]
mod inputs;
pub(crate) use inputs::*;
