use super::*;

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
            let here = super::models::imported(&c, id);
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
            let here = super::models::imported(&c, id);
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
