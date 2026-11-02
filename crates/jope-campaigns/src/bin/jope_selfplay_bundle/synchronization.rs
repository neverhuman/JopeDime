use super::*;

pub(super) fn sync_bundle_to_gpu_nodes(
    output_root: &Path,
    bundle_sha256: &str,
    manifest: &Value,
) -> Result<Value> {
    let source = output_root.join(bundle_sha256);
    let local_manifest_sha256 = sha256_file(&source.join("bundle.json"))?;
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("bundle manifest lacks file inventory for synchronization")?;
    let expected = rows
        .iter()
        .map(|row| {
            Ok((
                row.get("target")
                    .and_then(Value::as_str)
                    .context("bundle file lacks target")?,
                row.get("sha256")
                    .and_then(Value::as_str)
                    .context("bundle file lacks digest")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let hostname = local_hostname()?;
    let mut receipts = Vec::new();
    for node in GPU_SYNC_NODES {
        let (transport, observed) = if node_is_local(node, &hostname) {
            let mut observed = expected
                .iter()
                .map(|(target, _)| sha256_file(&source.join(target)))
                .collect::<Result<Vec<_>>>()?;
            observed.push(local_manifest_sha256.clone());
            ("local", observed)
        } else {
            let sync = std::process::Command::new("rsync")
                .arg("-a")
                .arg(&source)
                .arg(format!("{node}:{}/", output_root.display()))
                .output()
                .with_context(|| format!("spawn rsync of bundle {bundle_sha256} to {node}"))?;
            ensure!(
                sync.status.success(),
                "bundle sync to {node} failed: {}",
                String::from_utf8_lossy(&sync.stderr)
            );
            let mut command = std::process::Command::new("ssh");
            command.args(["-o", "BatchMode=yes", node, "sha256sum"]);
            for (target, _) in &expected {
                command.arg(source.join(target));
            }
            command.arg(source.join("bundle.json"));
            let check = command
                .output()
                .with_context(|| format!("spawn bundle sync verification on {node}"))?;
            ensure!(
                check.status.success(),
                "bundle {bundle_sha256} hash verification failed on {node}: {}",
                String::from_utf8_lossy(&check.stderr).trim()
            );
            let observed = String::from_utf8(check.stdout)?
                .lines()
                .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
                .collect::<Vec<_>>();
            ("ssh+rsync", observed)
        };
        ensure!(
            observed.len() == expected.len() + 1
                && observed
                    .iter()
                    .zip(expected.iter())
                    .all(|(actual, (_, digest))| actual == digest)
                && observed.last().map(String::as_str) == Some(local_manifest_sha256.as_str()),
            "bundle {bundle_sha256} failed complete asset verification on {node}"
        );
        receipts.push(json!({
            "node": node,
            "bundle_sha256": bundle_sha256,
            "bundle_manifest_sha256": local_manifest_sha256,
            "files_verified": expected.len(),
            "transport": transport,
            "verified": true,
        }));
    }
    Ok(json!({
        "schema": "JopeBundleSynchronizationReceipt.v1",
        "generated_at_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "bundle_sha256": bundle_sha256,
        "required_nodes": GPU_SYNC_NODES,
        "receipts": receipts,
        "complete": receipts.len() == GPU_SYNC_NODES.len(),
    }))
}

pub(super) fn node_command(
    node: &str,
    local_hostname: &str,
    program: &str,
    arguments: &[String],
) -> Result<std::process::Output> {
    let mut command = if node_is_local(node, local_hostname) {
        Command::new(program)
    } else {
        let mut command = Command::new("ssh");
        command.args(["-o", "BatchMode=yes", node, program]);
        command
    };
    command.args(arguments);
    command
        .output()
        .with_context(|| format!("run fresh activation probe on {node}: {program}"))
}

/// Repeat a challenged live probe during activation; each node hashes the
/// complete bundle again. See `tests::fresh_live_probe_records_rechallenged_hashes_for_all_nodes`.
pub(super) fn fresh_live_bundle_probe(
    output_root: &Path,
    bundle_sha256: &str,
    manifest: &Value,
) -> Result<Value> {
    fresh_live_bundle_probe_with(
        output_root,
        bundle_sha256,
        manifest,
        local_hostname,
        node_command,
    )
}

pub(super) fn fresh_live_bundle_probe_with<L, N>(
    output_root: &Path,
    bundle_sha256: &str,
    manifest: &Value,
    hostname: L,
    mut probe_node: N,
) -> Result<Value>
where
    L: FnOnce() -> Result<String>,
    N: FnMut(&str, &str, &str, &[String]) -> Result<std::process::Output>,
{
    let started_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let challenge_nonce_sha256 = sha256_bytes(
        format!(
            "jope-live-activation-probe-v2\0{bundle_sha256}\0{started_at_ms}\0{}",
            std::process::id()
        )
        .as_bytes(),
    );
    let root = output_root.join(bundle_sha256);
    let expected = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("bundle manifest lacks live-probe inventory")?
        .iter()
        .map(|row| {
            Ok((
                row.get("target")
                    .and_then(Value::as_str)
                    .context("live-probe target is absent")?
                    .to_owned(),
                row.get("sha256")
                    .and_then(Value::as_str)
                    .context("live-probe digest is absent")?
                    .to_owned(),
                row.get("bytes")
                    .and_then(Value::as_u64)
                    .context("live-probe size is absent")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let systemd = expected
        .iter()
        .filter(|(target, _, _)| {
            target.starts_with("runtime/systemd/")
                && (target.ends_with(".service") || target.ends_with(".timer"))
        })
        .map(|(target, digest, bytes)| json!({"target": target, "sha256": digest, "bytes": bytes}))
        .collect::<Vec<_>>();
    ensure!(
        !systemd.is_empty(),
        "live probe found no bundled systemd artifacts"
    );
    let manifest_file_sha256 = sha256_file(&root.join("bundle.json"))?;
    let manifest_canonical_json_sha256 = canonical_json_sha256(manifest)?;
    let hostname = hostname()?;
    let paths = expected
        .iter()
        .map(|(target, _, _)| root.join(target).display().to_string())
        .chain(std::iter::once(
            root.join("bundle.json").display().to_string(),
        ))
        .collect::<Vec<_>>();
    let mut receipts = Vec::new();
    for node in GPU_SYNC_NODES {
        let node_started_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let observed_hostname = probe_node(node, &hostname, "hostname", &[])?;
        ensure!(
            observed_hostname.status.success(),
            "fresh hostname probe failed on {node}"
        );
        let observed_hostname = String::from_utf8(observed_hostname.stdout)?
            .trim()
            .to_owned();
        ensure!(
            observed_hostname.split('.').next() == Some(*node),
            "fresh activation probe observed the wrong hostname for {node}"
        );
        let hashes = probe_node(node, &hostname, "sha256sum", &paths)?;
        ensure!(
            hashes.status.success(),
            "fresh bundle hash probe failed on {node}"
        );
        let observed = String::from_utf8(hashes.stdout.clone())?
            .lines()
            .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
            .collect::<Vec<_>>();
        ensure!(
            observed.len() == expected.len() + 1
                && observed
                    .iter()
                    .zip(expected.iter())
                    .all(|(actual, (_, digest, _))| actual == digest)
                && observed.last().map(String::as_str) == Some(manifest_file_sha256.as_str()),
            "fresh activation probe found incomplete or altered bytes on {node}"
        );
        let tool_path = "/usr/bin/sha256sum".to_owned();
        let tool_hash = probe_node(
            node,
            &hostname,
            "sha256sum",
            std::slice::from_ref(&tool_path),
        )?;
        ensure!(
            tool_hash.status.success(),
            "fresh probe tool hash failed on {node}"
        );
        let tool_sha256 = String::from_utf8(tool_hash.stdout.clone())?
            .split_whitespace()
            .next()
            .context("fresh probe tool digest is absent")?
            .to_owned();
        require_bundle_sha256(&tool_sha256)?;
        let version = probe_node(node, &hostname, "sha256sum", &["--version".to_owned()])?;
        ensure!(
            version.status.success(),
            "fresh probe tool version failed on {node}"
        );
        let version_text = String::from_utf8(version.stdout.clone())?;
        let raw_output_sha256 = sha256_bytes(
            &[
                observed_hostname.as_bytes(),
                &hashes.stdout,
                &hashes.stderr,
                &tool_hash.stdout,
                &version.stdout,
            ]
            .concat(),
        );
        receipts.push(json!({
            "node": node,
            "observed_hostname": observed_hostname,
            "challenge_nonce_sha256": challenge_nonce_sha256,
            "runtime_bundle_sha256": bundle_sha256,
            "bundle_manifest_file_sha256": manifest_file_sha256,
            "bundle_manifest_canonical_json_sha256": manifest_canonical_json_sha256,
            "complete_file_inventory": expected.iter().map(|(target, digest, bytes)| {
                json!({"target": target, "sha256": digest, "bytes": bytes})
            }).collect::<Vec<_>>(),
            "complete_systemd_artifact_inventory": systemd,
            "tool": {
                "path": tool_path,
                "executable_sha256": tool_sha256,
                "version": version_text.lines().next().unwrap_or_default(),
            },
            "started_at_unix_ms": node_started_at_ms,
            "finished_at_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "raw_output_sha256": raw_output_sha256,
        }));
    }
    ensure!(
        receipts.len() == GPU_SYNC_NODES.len(),
        "fresh five-node probe is incomplete"
    );
    let receipt = json!({
        "schema": "JopeLiveBundleProbe.v2",
        "challenge_nonce_sha256": challenge_nonce_sha256,
        "runtime_bundle_sha256": bundle_sha256,
        "generated_at_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "receipts": receipts,
    });
    let path = output_root
        .join("activation-probes")
        .join(format!("{bundle_sha256}-{challenge_nonce_sha256}.v2.json"));
    atomic_json(&path, &receipt)?;
    Ok(json!({
        "receipt": path,
        "receipt_file_sha256": sha256_file(&path)?,
        "challenge_nonce_sha256": challenge_nonce_sha256,
    }))
}

pub(super) fn local_hostname() -> Result<String> {
    let output = std::process::Command::new("hostname")
        .output()
        .context("read local hostname for bundle synchronization")?;
    ensure!(
        output.status.success(),
        "local hostname lookup failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let hostname = String::from_utf8(output.stdout)?.trim().to_owned();
    ensure!(
        !hostname.is_empty(),
        "local hostname lookup returned empty output"
    );
    Ok(hostname)
}

pub(super) fn node_is_local(node: &str, hostname: &str) -> bool {
    hostname.split('.').next() == Some(node)
}

pub(super) fn validate_no_python_runtime(contents: &str) -> Result<()> {
    let normalized = contents.to_ascii_lowercase();
    if normalized.contains("python") || normalized.contains(".py") {
        bail!("Python runtime reference is forbidden outside the PyTorch boundary");
    }
    Ok(())
}

pub(super) fn validate_encoding_runner(contents: &str) -> Result<()> {
    let normalized = contents.to_ascii_lowercase();
    ensure!(
        normalized.contains("jope_runtime_pythonpath")
            && normalized.contains("jope_encode_external_path")
            && normalized.contains("python3 \"$driver\"")
            && !normalized.contains(&format!(
                "export pythonpath={}",
                python_src_str().to_ascii_lowercase()
            ))
            && !normalized.contains(&default_encoding_driver_str().to_ascii_lowercase()),
        "encoding launcher escaped the bundled PyTorch boundary"
    );
    Ok(())
}
