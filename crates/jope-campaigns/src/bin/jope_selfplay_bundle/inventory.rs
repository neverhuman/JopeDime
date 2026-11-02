use super::*;

pub(super) fn inventory(candidates: &[Candidate]) -> Result<Vec<InventoryRow>> {
    candidates
        .iter()
        .map(|candidate| {
            Ok(InventoryRow {
                source: candidate.source.display().to_string(),
                target: candidate.target.clone(),
                sha256: sha256_file(&candidate.source)?,
                bytes: candidate.source.metadata()?.len(),
                python_role: candidate.python_role,
            })
        })
        .collect()
}

pub(super) fn prime_support_provenance(
    args: &Args,
    candidates: &[Candidate],
    inventory: &[InventoryRow],
) -> Result<PrimeSupportProvenance> {
    let source_bundle = args
        .prime
        .parent()
        .context("Prime capsule has no source bundle directory")?;
    ensure!(
        source_bundle.file_name().and_then(|name| name.to_str())
            == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "Prime support must come from immutable bundle {PRIME_SOURCE_BUNDLE_SHA256}"
    );
    let manifest_path = source_bundle.join("bundle.json");
    let manifest: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(&manifest_path)
            .with_context(|| format!("open Prime source bundle {}", manifest_path.display()))?,
    ))?;
    ensure!(
        manifest.get("bundle_sha256").and_then(Value::as_str) == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "Prime source bundle identity drift"
    );
    let source_rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("Prime source bundle lacks file inventory")?;
    let mut files = Vec::new();
    let mut seen = BTreeSet::new();
    for (candidate, row) in candidates.iter().zip(inventory) {
        if !candidate.source.starts_with(&args.prime) {
            continue;
        }
        let relative = candidate.source.strip_prefix(&args.prime)?;
        let source_target = format!("pytorch/{}", relative.display());
        let source_row = source_rows
            .iter()
            .find(|source| source.get("target").and_then(Value::as_str) == Some(&source_target))
            .with_context(|| format!("Prime source bundle lacks {source_target}"))?;
        ensure!(
            source_row.get("sha256").and_then(Value::as_str) == Some(row.sha256.as_str())
                && source_row.get("bytes").and_then(Value::as_u64) == Some(row.bytes),
            "Prime source bundle content drift for {source_target}"
        );
        if seen.insert((source_target.clone(), row.target.clone())) {
            files.push(PrimeSupportFile {
                source_bundle_target: source_target,
                bundle_target: row.target.clone(),
                sha256: row.sha256.clone(),
                bytes: row.bytes,
            });
        }
    }
    ensure!(!files.is_empty(), "Prime support capsule is empty");
    files.sort_by(|left, right| {
        left.source_bundle_target
            .cmp(&right.source_bundle_target)
            .then_with(|| left.bundle_target.cmp(&right.bundle_target))
    });
    let package_tree_sha256 = prime_support_identity(&files)?;
    Ok(PrimeSupportProvenance {
        source_bundle_sha256: PRIME_SOURCE_BUNDLE_SHA256,
        source_bundle_manifest_sha256: sha256_file(&manifest_path)?,
        package_tree_sha256,
        files,
    })
}

pub(super) fn prime_support_identity(files: &[PrimeSupportFile]) -> Result<String> {
    let rows = files
        .iter()
        .map(|row| {
            json!([
                row.source_bundle_target,
                row.bundle_target,
                row.sha256,
                row.bytes,
            ])
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&serde_json::to_vec(&rows)?))
}

pub(super) fn inventory_identity(inventory: &[InventoryRow]) -> Result<String> {
    let rows: Vec<Value> = inventory
        .iter()
        .map(|row| {
            json!({
                "target": row.target,
                "sha256": row.sha256,
                "bytes": row.bytes,
                "python_role": row.python_role,
            })
        })
        .collect();
    Ok(sha256_bytes(&serde_json::to_vec(&rows)?))
}

pub(super) fn require_bundle_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "bundle identity is not a lowercase SHA-256"
    );
    Ok(())
}

pub(super) fn manifest_inventory_identity(rows: &[Value]) -> Result<String> {
    let identity_rows = rows
        .iter()
        .map(|row| {
            json!({
                "target": row.get("target").cloned().unwrap_or(Value::Null),
                "sha256": row.get("sha256").cloned().unwrap_or(Value::Null),
                "bytes": row.get("bytes").cloned().unwrap_or(Value::Null),
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&serde_json::to_vec(&identity_rows)?))
}
