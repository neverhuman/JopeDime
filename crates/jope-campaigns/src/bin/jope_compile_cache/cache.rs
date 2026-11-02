use super::*;

pub(super) fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .context("output filename is not UTF-8")?,
        std::process::id()
    ));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(super) fn write_operation(
    args: &Args,
    request_sha256: &str,
    idempotency_key: &str,
    dataset: &Dataset,
    candidate: &Candidate,
    cache_key: &str,
    hit: bool,
) -> Result<PathBuf> {
    let identity = json!({
        "request_sha256": request_sha256,
        "idempotency_key": idempotency_key,
        "cache_key": cache_key,
    });
    let operation_sha256 = sha256_bytes(&canonical_json_bytes(&identity)?);
    let path = cache_root(args)
        .join("operations")
        .join(format!("{operation_sha256}.json"));
    let value = json!({
        "schema": OPERATION_SCHEMA,
        "operation_sha256": operation_sha256,
        "request_sha256": request_sha256,
        "idempotency_key": idempotency_key,
        "cache_key": cache_key,
        "dataset_hash": dataset.dataset_hash,
        "semantic_sha256": candidate.semantic_sha256,
        "outcome": if hit { "hit" } else { "miss" },
        "created_at_ms": now_ms(),
    });
    if path.is_file() {
        let prior = read_json(&path)?;
        for field in [
            "schema",
            "operation_sha256",
            "request_sha256",
            "idempotency_key",
            "cache_key",
            "dataset_hash",
            "semantic_sha256",
        ] {
            ensure!(prior[field] == value[field], "idempotency receipt drift");
        }
        return Ok(path);
    }
    atomic_json(&path, &value)?;
    Ok(path)
}

pub(super) fn cache_root(args: &Args) -> PathBuf {
    args.root.join("compiled-cache")
}

pub(super) fn entry_root(args: &Args, key: &str) -> PathBuf {
    cache_root(args).join("entries").join(&key[..2]).join(key)
}

pub(super) fn cache_status(root: &Path) -> Result<Value> {
    let mut entries = 0usize;
    let mut valid_entries = 0usize;
    let entries_root = root.join("entries");
    if entries_root.is_dir() {
        for prefix in fs::read_dir(&entries_root)? {
            let prefix = prefix?.path();
            if !prefix.is_dir() {
                continue;
            }
            for entry in fs::read_dir(prefix)? {
                let manifest = entry?.path().join("artifact-manifest.json");
                if !manifest.is_file() {
                    continue;
                }
                entries += 1;
                if generic_entry_valid(&manifest).unwrap_or(false) {
                    valid_entries += 1;
                }
            }
        }
    }
    let mut hits = 0usize;
    let mut misses = 0usize;
    let operations = root.join("operations");
    if operations.is_dir() {
        for row in fs::read_dir(operations)? {
            let path = row?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let value = match read_json(&path) {
                Ok(value) if value["schema"] == OPERATION_SCHEMA => value,
                _ => continue,
            };
            match value["outcome"].as_str() {
                Some("hit") => hits += 1,
                Some("miss") => misses += 1,
                _ => {}
            }
        }
    }
    let requests = hits + misses;
    Ok(json!({
        "entries": entries,
        "valid_entries": valid_entries,
        "invalid_entries": entries - valid_entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": if requests == 0 { Value::Null } else { json!(hits as f64 / requests as f64) },
    }))
}

pub(super) fn generic_entry_valid(manifest_path: &Path) -> Result<bool> {
    let manifest = read_json(manifest_path)?;
    if manifest["schema"] != ARTIFACT_SCHEMA
        || manifest["prediction_bit_parity"] != true
        || manifest["score_bit_parity"] != true
        || manifest["qualified"] != true
    {
        return Ok(false);
    }
    let entry = manifest_path.parent().context("manifest has no parent")?;
    let relative = manifest["artifact"]
        .as_str()
        .context("artifact is missing")?;
    let expected = manifest["artifact_sha256"]
        .as_str()
        .context("artifact hash is missing")?;
    Ok(relative == "fitted-model.codec" && sha256_file(&entry.join(relative))? == expected)
}
