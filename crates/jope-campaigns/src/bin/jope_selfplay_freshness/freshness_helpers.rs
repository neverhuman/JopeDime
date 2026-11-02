use super::*;

pub(super) fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub(super) fn freshness_scoped_manifest(manifest: &Value) -> bool {
    if manifest.get("schema").and_then(Value::as_str) != Some("JopeFlywheelCorpusEvalShard.v1")
        || !manifest.get("candidates").is_some_and(Value::is_array)
    {
        return false;
    }
    let lane = manifest
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    !(lane.contains("gate") || lane.contains("sealed") || lane.contains("canary"))
}

pub(super) fn manifest_epoch(manifest: &Value) -> String {
    if let Some(epoch) = manifest
        .get("checkpoint_epoch")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        return epoch.to_owned();
    }
    let checkpoint = manifest
        .get("label_authority")
        .and_then(|authority| authority.get("model_checkpoint_sha256"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if is_sha256(checkpoint) {
        return format!("checkpoint-{checkpoint}");
    }
    "legacy-jope-testonly".to_owned()
}

pub(super) fn epoch_for_checkpoint(checkpoint: &str) -> Result<String> {
    if !is_sha256(checkpoint) {
        bail!("checkpoint epoch requires a SHA-256 checkpoint identity");
    }
    Ok(format!("checkpoint-{checkpoint}"))
}

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(super) fn content_hash(value: &Value) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

pub(super) fn auto_retirement_idempotency_key(epoch: &str, preview_sha256: &str) -> Result<String> {
    if epoch.is_empty() || !is_sha256(preview_sha256) {
        bail!("automatic retirement lacks deterministic provenance");
    }
    Ok(format!(
        "auto-retirement-{}",
        content_hash(&json!({
            "schema": "JopeAutomaticStaleRetirementIdempotency.v1",
            "active_epoch": epoch,
            "preview_receipt_sha256": preview_sha256
        }))?
    ))
}

pub(super) fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub(super) fn null_if_empty(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        json!(value)
    }
}

pub(super) fn retirement_scoped_job(job: &InventoryJob) -> bool {
    job.outside_scq
        || job
            .scheduler
            .project
            .as_deref()
            .is_some_and(|project| RETIRABLE_PROJECTS.contains(&project))
}
