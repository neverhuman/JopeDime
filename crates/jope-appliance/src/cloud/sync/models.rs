use super::*;

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
