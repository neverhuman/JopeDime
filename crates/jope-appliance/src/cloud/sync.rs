use super::*;

pub use telemetry::sync_events;
pub(super) use telemetry::{record_publication, telemetry_command};
pub(super) use upload::check_upload;
pub use upload::{publish_file, Upload};

mod telemetry;
mod upload;

#[derive(Serialize, Deserialize)]
pub(super) struct SyncRecord {
    origin: String,
    account_id: String,
    local_run_id: String,
    sequence: i64,
    snapshot_hash: String,
    body: Value,
    cloud_id: Option<String>,
    cloud_url: Option<String>,
    synced: bool,
}
fn uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(n, b)| {
            if [8, 13, 18, 23].contains(&n) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}
fn account_root(c: &Connection) -> Result<PathBuf> {
    anyhow::ensure!(uuid(&c.account_id), "invalid stored account identity");
    let name = format!(
        "{}-{}",
        &jope_runtime::sha256_hex(c.origin.as_bytes())[..16],
        c.account_id
    );
    let root = cloud_root()?.join("accounts").join(name);
    private_dir(&root)?;
    Ok(root)
}
fn device_unlocked() -> Result<String> {
    let path = cloud_root()?.join("device-id");
    if path.exists() {
        let value = String::from_utf8(read_private(&path)?)?;
        anyhow::ensure!(uuid(&value), "invalid device identity");
        return Ok(value);
    }
    let hex = format!("{:032x}", ulid::Ulid::new().0);
    let id = format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    write_private(&path, id.as_bytes())?;
    Ok(id)
}
fn device_id() -> Result<String> {
    let _lock = connection_lock()?;
    device_unlocked()
}
fn binding(c: &Connection, run: &str) -> Result<()> {
    let path = cloud_root()?.join("bindings").join(format!("{run}.json"));
    if path.exists() {
        let value: Value = serde_json::from_slice(&read_private(&path)?)?;
        anyhow::ensure!(
            value["accountId"] == c.account_id && value["origin"] == c.origin,
            "this local run is bound to another VEOX account or workspace; it was not uploaded"
        );
    } else {
        write_private(
            &path,
            &serde_json::to_vec(&json!({"accountId":c.account_id,"origin":c.origin}))?,
        )?;
    }
    Ok(())
}
fn record_path(c: &Connection, run: &str) -> Result<PathBuf> {
    Ok(account_root(c)?.join("runs").join(format!("{run}.json")))
}
pub fn local_client() -> Result<jope_client::Http> {
    use jope_client::discover::{self, Engine, Mode, Options};
    match discover::discover(&Options {
        home: jope_paths::jopedime_home(),
        executable: std::env::current_exe()?,
        endpoint: None,
        mode: Mode::Interactive,
        engine_available: cfg!(feature = "engine"),
    })? {
        Engine::Background(engine) => Ok(engine.client),
        _ => anyhow::bail!("a local background engine is required for this action"),
    }
}
impl Client {
    fn connected_records(&self) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        let mut after = None;
        let mut seen = std::collections::HashSet::new();
        loop {
            let path = after
                .as_ref()
                .map(|v: &String| format!("/api/cli/runs?after={}", encode(v)))
                .unwrap_or_else(|| "/api/cli/runs".into());
            let page = self.json("GET", &path, Value::Null)?;
            let rows = page["runs"]
                .as_array()
                .context("invalid connected run page")?;
            result.extend(rows.iter().cloned());
            match page["nextAfter"].as_str() {
                None => return Ok(result),
                Some(cursor) => {
                    anyhow::ensure!(
                        uuid(cursor) && seen.insert(cursor.to_owned()),
                        "invalid or repeated cloud cursor"
                    );
                    after = Some(cursor.to_owned());
                }
            }
        }
    }
    fn cloud_records(&self) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        for archived in [false, true] {
            let mut older: Option<String> = None;
            let mut seen = std::collections::HashSet::new();
            loop {
                let mut path = format!(
                    "/api/sessions?limit=100{}",
                    if archived { "&archived=1" } else { "" }
                );
                if let Some(cursor) = &older {
                    path.push_str(&format!("&before={}", encode(cursor)));
                }
                let page = self.json("GET", &path, Value::Null)?;
                for row in page["runs"].as_array().context("invalid cloud run page")? {
                    let mut row = row.clone();
                    row["archived"] = json!(archived);
                    result.push(row);
                }
                match page["older"].as_str() {
                    None => break,
                    Some(cursor) => {
                        anyhow::ensure!(
                            seen.insert(cursor.to_owned()),
                            "cloud repeated a page cursor"
                        );
                        older = Some(cursor.to_owned());
                    }
                }
            }
        }
        Ok(result)
    }
}
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
pub fn publish_model(run: &str, model: &Path) -> Result<String> {
    use jope_client::Transport;
    let id: jope_api::ids::RunId = run.parse()?;
    let local = local_client()?;
    let snapshot = local.run_status(&id)?;
    let version = local.system_info()?.version;
    sync_snapshot(&snapshot, &version)?;
    let object = jope_runtime::model_object::ModelObject::read(model, MAX_MODEL)?;
    let bytes = pack_object(&object)?;
    let bytes_len = bytes.len() as u64;
    let hash = jope_runtime::sha256_hex(&bytes);
    let _lock = connection_lock()?;
    let c = connection()?;
    binding(&c, run)?;
    let record: SyncRecord = serde_json::from_slice(&read_private(&record_path(&c, run)?)?)?;
    let cloud = record.cloud_id.context("run is not synced")?;
    anyhow::ensure!(uuid(&cloud), "invalid cloud run");
    let client = Client::connected(&c)?;
    let mut request = client.wire(
        "POST",
        &format!("/api/cli/runs/{cloud}/model"),
        bytes,
        "application/zip",
    )?;
    request
        .headers_mut()
        .insert("x-model-sha256", hash.parse()?);
    let mut response = client.agent.run(request)?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_JSON + 1)
        .read_to_end(&mut body)?;
    anyhow::ensure!(status==200,"model publication refused (HTTP {status}); an existing different published model is immutable");
    let value: Value = serde_json::from_slice(&body)?;
    anyhow::ensure!(
        value["modelSha256"] == hash,
        "published model transfer digest differs"
    );
    let note = record_publication(&c, &client, run, &cloud, &hash, bytes_len)
        .unwrap_or_else(|error| format!("\nModel event queued: {error:#}"));
    Ok(format!(
        "Published model {} · SHA-256 {hash}\n{}/runs/{cloud}{note}",
        object.id(),
        c.origin
    ))
}
fn imported(c: &Connection, cloud: &str) -> bool {
    let result = (|| -> Result<()> {
        anyhow::ensure!(uuid(cloud), "invalid run");
        let path = account_root(c)?
            .join("imports")
            .join(format!("{cloud}.json"));
        let record: Value = serde_json::from_slice(&read_private(&path)?)?;
        anyhow::ensure!(
            record["accountId"] == c.account_id && record["origin"] == c.origin,
            "different account import"
        );
        let path = record["modelPath"]
            .as_str()
            .context("import has no model")?;
        jope_runtime::model_object::ModelObject::read(Path::new(path), MAX_MODEL)?;
        Ok(())
    })();
    result.is_ok()
}
pub fn fetch_model(cloud: &str) -> Result<String> {
    anyhow::ensure!(uuid(cloud), "cloud run must be a UUID");
    let c = connection()?;
    let client = Client::connected(&c)?;
    let connected = client.call(
        "GET",
        &format!("/api/cli/runs/{cloud}"),
        Vec::new(),
        "application/json",
        MAX_JSON,
    )?;
    let (path, expected) = if connected.0 == 200 {
        let row: Value = serde_json::from_slice(&connected.1)?;
        (
            format!("/api/cli/runs/{cloud}/model"),
            Some(
                row["modelSha256"]
                    .as_str()
                    .context("no model has been published")?
                    .to_owned(),
            ),
        )
    } else if connected.0 == 404 {
        (
            format!("/api/sessions/{cloud}/artifacts/model_object"),
            None,
        )
    } else {
        anyhow::bail!("could not read cloud run (HTTP {})", connected.0)
    };
    let (status, bytes) = client.call("GET", &path, Vec::new(), "application/json", MAX_MODEL)?;
    anyhow::ensure!(status == 200, "model download unavailable (HTTP {status})");
    if let Some(expected) = expected {
        anyhow::ensure!(
            jope_runtime::sha256_hex(&bytes) == expected,
            "model download digest mismatch"
        );
    }
    let object = unpack_object(&bytes)?;
    let root = account_root(&c)?;
    let models = root.join("models");
    private_dir(&models)?;
    let path = object.save(&models, MAX_MODEL)?;
    write_private(
        &root.join("imports").join(format!("{cloud}.json")),
        &serde_json::to_vec(
            &json!({"accountId":c.account_id,"origin":c.origin,"modelPath":path,"objectId":object.id()}),
        )?,
    )?;
    Ok(format!(
        "Downloaded and verified {}\nModel: {}\nExecutable here: {}",
        object.id(),
        path.display(),
        object.verify_executable().is_ok()
    ))
}
pub fn run_entries(
    local: &jope_client::Http,
) -> Result<(Vec<crate::terminal::runs::Entry>, String)> {
    use crate::terminal::runs::Entry;
    let snapshots = crate::terminal::runs::local_runs(local)?;
    let mut entries: Vec<Entry> = snapshots
        .into_iter()
        .map(|run| Entry {
            id: run.run_id.to_string(),
            status: format!("{:?}", run.state).to_lowercase(),
            local: Some(run.run_id),
            cloud_url: None,
            model_here: run.best.is_some(),
            model_cloud: false,
            local_record: true,
        })
        .collect();
    let c = match connection() {
        Ok(c) => c,
        Err(_) => return Ok((entries, "Local engine · /login to connect VEOX".into())),
    };
    let client = Client::connected(&c)?;
    let device = device_id()?;
    let result = (|| -> Result<()> {
        for row in client.connected_records()? {
            let id = row["id"]
                .as_str()
                .filter(|id| uuid(id))
                .context("invalid cloud identity")?;
            let url = format!("{}/runs/{id}", c.origin);
            anyhow::ensure!(row["url"] == url, "invalid cloud URL");
            let local_id = row["metadata"]["localRunId"].as_str().unwrap_or("");
            if row["metadata"]["deviceId"] == device {
                if let Some(entry) = entries.iter_mut().find(|entry| entry.id == local_id) {
                    entry.cloud_url = Some(url);
                    entry.model_cloud = row["modelSha256"].is_string();
                    continue;
                }
            }
            let here = imported(&c, id);
            entries.push(Entry {
                id: id.into(),
                status: row["metadata"]["status"]
                    .as_str()
                    .unwrap_or("unknown")
                    .into(),
                local: None,
                cloud_url: Some(url),
                model_here: here,
                model_cloud: row["modelSha256"].is_string(),
                local_record: here,
            });
        }
        for row in client.cloud_records()? {
            let id = row["id"]
                .as_str()
                .filter(|id| uuid(id))
                .context("invalid cloud identity")?;
            let here = imported(&c, id);
            entries.push(Entry {
                id: id.into(),
                status: format!(
                    "{}{}",
                    row["state"].as_str().unwrap_or("unknown"),
                    if row["archived"] == true {
                        " (archived)"
                    } else {
                        ""
                    }
                ),
                local: None,
                cloud_url: Some(format!("{}/runs/{id}", c.origin)),
                model_here: here,
                model_cloud: row["hasModel"].as_bool().unwrap_or(false),
                local_record: here,
            });
        }
        Ok(())
    })();
    let notice = match result {
        Ok(()) => format!("{} · {}", c.email, c.origin),
        Err(error) => format!("Cloud refresh incomplete: {error:#}"),
    };
    Ok((entries, notice))
}

pub fn publish_best(run: &str) -> Result<String> {
    use jope_client::Transport;
    let local = local_client()?;
    let id: jope_api::ids::RunId = run.parse()?;
    let snapshot = local.run_status(&id)?;
    let version = local.system_info()?.version;
    sync_snapshot(&snapshot, &version)?;
    let best = snapshot
        .best
        .context("the run has no fitted model to publish")?;
    let request = jope_api::ExportRequest {
        algorithm_id: best.algorithm_id,
        kind: jope_api::ExportKind::Model,
        languages: Vec::new(),
        require_all_languages: false,
        refit: jope_api::Refit::FitHoldout,
    };
    let key = format!("cloud_publish_{}", ulid::Ulid::new());
    let export = local.model_export(&id, &request, Some(&key))?;
    let response = local.artifact_download(&export.artifact_id)?;
    let expected = response
        .headers
        .get("x-jope-artifact-sha256")
        .and_then(|v| v.to_str().ok())
        .context("export has no transfer digest")?
        .to_owned();
    let mut bytes = Vec::new();
    response.body.take(MAX_MODEL + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_MODEL && jope_runtime::sha256_hex(&bytes) == expected,
        "local export integrity check failed"
    );
    let object = unpack_object(&bytes)?;
    let c = connection()?;
    let models = account_root(&c)?.join("models");
    private_dir(&models)?;
    let model = object.save(&models, MAX_MODEL)?;
    publish_model(run, &model)
}
