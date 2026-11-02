use super::*;

pub fn sync_snapshot(snapshot: &jope_api::RunSnapshot, version: &str) -> Result<String> {
    sync_snapshot_for(snapshot, version, None)
}
pub fn sync_snapshot_for(
    snapshot: &jope_api::RunSnapshot,
    version: &str,
    expected: Option<&(String, String)>,
) -> Result<String> {
    let _lock = connection_lock()?;
    let c = connection()?;
    if let Some((origin, account)) = expected {
        anyhow::ensure!(
            &c.origin == origin && &c.account_id == account,
            "VEOX account changed; automatic synchronization stopped"
        );
    }
    anyhow::ensure!(
        c.may_publish,
        "this token is read-only; reconnect with CLI publication access"
    );
    let run = snapshot.run_id.to_string();
    binding(&c, &run)?;
    let device = device_unlocked()?;
    let path = record_path(&c, &run)?;
    let value = serde_json::to_value(snapshot)?;
    let wall = &value["budget_used"]["wall_seconds"];
    let mut body = json!({"deviceId":device,"localRunId":run,"label":run,"status":value["state"],"phase":value["phase"],"engineVersion":version,"revision":snapshot.revision.0.to_string(),"eventCursor":snapshot.event_watermark.0.to_string(),"evaluations":snapshot.budget_used.evaluations.0.to_string(),"elapsedSeconds":wall.as_str().map(str::to_owned).unwrap_or_else(||wall.to_string()),"metricName":snapshot.best.as_ref().map(|b|b.metric.clone()),"bestMetric":snapshot.best.as_ref().map(|b|b.cv_value).filter(|v|v.is_finite())});
    let fingerprint = jope_runtime::sha256_hex(&serde_json::to_vec(&body)?);
    let previous = if path.exists() {
        Some(serde_json::from_slice::<SyncRecord>(&read_private(&path)?)?)
    } else {
        None
    };
    let mut record = if let Some(previous) = previous.filter(|r| r.snapshot_hash == fingerprint) {
        previous
    } else {
        let sequence = if path.exists() {
            serde_json::from_slice::<SyncRecord>(&read_private(&path)?)?
                .sequence
                .checked_add(1)
                .context("metadata sequence exhausted")?
        } else {
            1
        };
        body["sequence"] = json!(sequence.to_string());
        SyncRecord {
            origin: c.origin.clone(),
            account_id: c.account_id.clone(),
            local_run_id: run.clone(),
            sequence,
            snapshot_hash: fingerprint,
            body,
            cloud_id: None,
            cloud_url: None,
            synced: false,
        }
    };
    anyhow::ensure!(
        record.origin == c.origin && record.account_id == c.account_id,
        "outbox belongs to another account"
    );
    if record.synced {
        return Ok(format!(
            "Up to date: {}",
            record.cloud_url.unwrap_or_default()
        ));
    }
    write_private(&path, &serde_json::to_vec(&record)?)?;
    let client = Client::connected(&c)?;
    for attempt in 0..2 {
        let (status, bytes) = client
            .call(
                "POST",
                "/api/cli/runs",
                serde_json::to_vec(&record.body)?,
                "application/json",
                MAX_JSON,
            )
            .context("metadata is queued locally; connection failed")?;
        if status == 409 && attempt == 0 {
            let remote = client
                .connected_records()?
                .into_iter()
                .find(|r| r["metadata"]["deviceId"] == device && r["metadata"]["localRunId"] == run)
                .context("cloud rejected metadata order; refresh the run list")?;
            let sequence = remote["sequence"]
                .as_str()
                .context("invalid cloud sequence")?
                .parse::<i64>()?;
            record.sequence = record
                .sequence
                .max(sequence)
                .checked_add(1)
                .context("metadata sequence exhausted")?;
            record.body["sequence"] = json!(record.sequence.to_string());
            write_private(&path, &serde_json::to_vec(&record)?)?;
            continue;
        }
        anyhow::ensure!(
            status == 200,
            "metadata is queued locally; cloud refused it (HTTP {status})"
        );
        let response: Value = serde_json::from_slice(&bytes)?;
        let id = response["id"]
            .as_str()
            .context("cloud returned no run identity")?;
        anyhow::ensure!(uuid(id), "cloud returned invalid run identity");
        let url = format!("{}/runs/{id}", c.origin);
        anyhow::ensure!(
            response["url"] == url,
            "cloud returned an unexpected run URL"
        );
        record.cloud_id = Some(id.to_owned());
        record.cloud_url = Some(url.clone());
        record.synced = true;
        write_private(&path, &serde_json::to_vec(&record)?)?;
        return Ok(format!("Synced {run}: {url}"));
    }
    anyhow::bail!("metadata remains queued locally")
}
pub(super) fn pack_object(object: &jope_runtime::model_object::ModelObject) -> Result<Vec<u8>> {
    let total =
        object.manifest_bytes().len() + object.parts().values().map(Vec::len).sum::<usize>();
    anyhow::ensure!(total as u64 <= MAX_MODEL, "model exceeds 32 MiB");
    let mut zip = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("manifest.json", options)?;
    zip.write_all(object.manifest_bytes())?;
    for (name, bytes) in object.parts() {
        zip.start_file(name, options)?;
        zip.write_all(bytes)?;
    }
    let bytes = zip.finish()?.into_inner();
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_MODEL,
        "model archive exceeds 32 MiB"
    );
    Ok(bytes)
}
pub(super) fn unpack_object(bytes: &[u8]) -> Result<jope_runtime::model_object::ModelObject> {
    let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))?;
    let nested = archive.by_name("model/manifest.json").is_ok();
    anyhow::ensure!(archive.len() <= 260, "model has too many parts");
    let mut parts = std::collections::BTreeMap::new();
    let mut manifest = None;
    let mut total = 0u64;
    // Export bundles use model/<flat filename>; connected model bundles are flat.
    for n in 0..archive.len() {
        let mut file = archive.by_index(n)?;
        let name = file.name()?.into_owned();
        anyhow::ensure!(
            !name.contains('\\')
                && !name.starts_with('/')
                && !name.split('/').any(|p| p == "..")
                && !file.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000),
            "unsafe model archive path"
        );
        if file.is_dir() || (nested && !name.starts_with("model/")) {
            continue;
        }
        let name = if let Some(name) = name.strip_prefix("model/") {
            name.to_owned()
        } else {
            name
        };
        if matches!(
            name.as_str(),
            "export.json" | "parity_receipt.json" | "receipt.json" | "README.md"
        ) {
            continue;
        }
        anyhow::ensure!(!name.contains('/'), "model archive is not flat");
        total = total
            .checked_add(file.size())
            .context("model byte limit exceeded")?;
        anyhow::ensure!(total <= MAX_MODEL, "model expands beyond 32 MiB");
        let mut data = Vec::new();
        file.by_ref().take(MAX_MODEL + 1).read_to_end(&mut data)?;
        anyhow::ensure!(data.len() as u64 == file.size(), "model part size mismatch");
        if name == "manifest.json" {
            anyhow::ensure!(manifest.replace(data).is_none(), "duplicate model manifest");
        } else {
            anyhow::ensure!(parts.insert(name, data).is_none(), "duplicate model part");
        }
    }
    jope_runtime::model_object::ModelObject::decode(
        manifest.context("model manifest missing")?,
        parts,
        MAX_MODEL,
    )
}
