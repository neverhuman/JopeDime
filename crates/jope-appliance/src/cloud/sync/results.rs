//! After a local run finishes: its results go to the workspace so the run can be explored there
//! like a cloud run (predictions from the fitted model, downloads, generated code).
//!
//! Order is the contract's (ai-veox-app `docs/connected-cli.md`, *Local runs are first-class*):
//! metadata (`POST /api/cli/runs`, which names the cloud run), events (`/:id/events`), the fitted
//! model (`/:id/model`), the generated code (`/:id/code`), then `model_published` in a last events
//! batch — after the model, which the event is checked against. Each upload is retried on a
//! transport failure, a 5xx or a 429; what succeeded is remembered per run, so the next sync sends
//! only what is still missing. Exact retries are idempotent on the server; a different model for
//! the same run is refused. The report PDF is written by the engine on a cloud run; a local run
//! uploads one only when the engine writes it (not yet), so none is sent here.
//! No dataset rows are sent — but a kernel, smoother or Gaussian-process model keeps the
//! support rows it predicts from inside its fitted state, which is why the consent says so.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Model,
    Code,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Code => "code",
        }
    }
    fn wire(self) -> (&'static str, &'static str, &'static str, u64) {
        match self {
            Self::Model => (
                "application/zip",
                "x-model-sha256",
                "modelSha256",
                MAX_MODEL,
            ),
            Self::Code => ("application/zip", "x-code-sha256", "codeSha256", 8 << 20),
        }
    }
}

/// What has reached the workspace for one run.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct Delivered {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

impl Delivered {
    fn slot(&mut self, kind: Kind) -> &mut Option<String> {
        match kind {
            Kind::Model => &mut self.model,
            Kind::Code => &mut self.code,
        }
    }
}

/// One attempt's outcome.
pub(crate) enum Attempt<T> {
    Done(T),
    Retry(String),
    Fail(String),
}

/// Run `op` up to `attempts` times, sleeping 1 s, 2 s, 4 s… between retryable failures.
pub(crate) fn retry<T>(
    attempts: u32,
    sleep: &dyn Fn(Duration),
    mut op: impl FnMut(u32) -> Attempt<T>,
) -> std::result::Result<T, String> {
    let mut last = String::new();
    for n in 0..attempts {
        match op(n) {
            Attempt::Done(value) => return Ok(value),
            Attempt::Fail(message) => return Err(message),
            Attempt::Retry(message) => {
                last = message;
                if n + 1 < attempts {
                    sleep(Duration::from_secs(1 << n.min(4)));
                }
            }
        }
    }
    Err(last)
}

fn send(client: &Client, cloud: &str, kind: Kind, bytes: &[u8], hash: &str) -> Attempt<()> {
    let (media, header, field, _) = kind.wire();
    let request = match client
        .wire(
            "POST",
            &format!("/api/cli/runs/{cloud}/{}", kind.label()),
            bytes.to_vec(),
            media,
        )
        .and_then(|mut r| {
            r.headers_mut().insert(header, hash.parse()?);
            Ok(r)
        }) {
        Ok(request) => request,
        Err(error) => return Attempt::Fail(format!("{error:#}")),
    };
    let mut response = match client.agent.run(request) {
        Ok(response) => response,
        Err(error) => return Attempt::Retry(format!("connection failed: {error}")),
    };
    let status = response.status().as_u16();
    let mut body = Vec::new();
    let _ = response
        .body_mut()
        .as_reader()
        .take(MAX_JSON + 1)
        .read_to_end(&mut body);
    match status {
        200 => {
            let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            if value[field] == hash {
                Attempt::Done(())
            } else {
                Attempt::Fail(format!(
                    "the workspace recorded a different {} digest",
                    kind.label()
                ))
            }
        }
        429 | 500..=599 => Attempt::Retry(format!("HTTP {status}")),
        409 => Attempt::Fail(format!(
            "a different {} is already published for this run (published results are immutable)",
            kind.label()
        )),
        _ => {
            let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            match super::route::Limit::parse(status, &value) {
                Some(limit) => Attempt::Fail(limit.sentence()),
                None => Attempt::Fail(format!("refused (HTTP {status})")),
            }
        }
    }
}

fn delivered_model(state: &Path) -> Option<String> {
    let delivered: Delivered = serde_json::from_slice(&read_private(state).ok()?).ok()?;
    delivered.model
}

/// Upload `items` in order to the cloud run, skipping what `state` says already arrived.
pub(crate) fn push(
    client: &Client,
    cloud: &str,
    items: Vec<(Kind, Vec<u8>)>,
    state: &Path,
    sleep: &dyn Fn(Duration),
) -> Vec<String> {
    let mut delivered: Delivered = read_private(state)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let mut lines = Vec::new();
    for (kind, bytes) in items {
        let (_, _, _, limit) = kind.wire();
        if bytes.is_empty() || bytes.len() as u64 > limit {
            lines.push(format!(
                "Results: the {} is empty or too large to sync; skipped.",
                kind.label()
            ));
            continue;
        }
        let hash = jope_runtime::sha256_hex(&bytes);
        if delivered.slot(kind).as_deref() == Some(hash.as_str()) {
            continue;
        }
        match retry(3, sleep, |_| send(client, cloud, kind, &bytes, &hash)) {
            Ok(()) => {
                *delivered.slot(kind) = Some(hash);
                let _ = write_private(state, &serde_json::to_vec(&delivered).unwrap_or_default());
                lines.push(format!("Results: {} synced.", kind.label()));
            }
            Err(error) => lines.push(format!(
                "Results: the {} did not sync ({error}); it is retried on the next sync.",
                kind.label()
            )),
        }
    }
    lines
}

fn export<T: jope_client::Transport>(
    local: &T,
    run: &jope_api::ids::RunId,
    request: &jope_api::ExportRequest,
    limit: u64,
) -> Result<Vec<u8>> {
    let key = format!("cloud_results_{}", ulid::Ulid::new());
    let export = local.model_export(run, request, Some(&key))?;
    let response = local.artifact_download(&export.artifact_id)?;
    let expected = response
        .headers
        .get("x-jope-artifact-sha256")
        .and_then(|v| v.to_str().ok())
        .context("export has no transfer digest")?
        .to_owned();
    let mut bytes = Vec::new();
    response.body.take(limit + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= limit && jope_runtime::sha256_hex(&bytes) == expected,
        "local export integrity check failed"
    );
    Ok(bytes)
}

/// A generated-code export flattened to the workspace's code archive (flat files only).
pub(crate) fn flatten_code(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))?;
    let mut out = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut names = std::collections::BTreeSet::new();
    let mut total = 0u64;
    for n in 0..archive.len() {
        let mut file = archive.by_index(n)?;
        let path = file.name()?.into_owned();
        if file.is_dir() || path.contains('\\') || path.split('/').any(|p| p == "..") {
            continue;
        }
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let Some(base) = parts.last() else { continue };
        if base.starts_with('.') {
            continue;
        }
        let mut name = (*base).to_owned();
        if !names.insert(name.clone()) {
            name = parts.join("_");
            if !names.insert(name.clone()) {
                continue;
            }
        }
        total += file.size();
        anyhow::ensure!(total <= 8 << 20, "generated code exceeds 8 MiB");
        let mut data = Vec::new();
        file.by_ref().take(8 << 20).read_to_end(&mut data)?;
        out.start_file(name, options)?;
        out.write_all(&data)?;
    }
    anyhow::ensure!(
        !names.is_empty() && names.len() <= 256,
        "no generated code to sync"
    );
    Ok(out.finish()?.into_inner())
}

/// Sync a finished local run's results: metadata and events, then model and code, then
/// `model_published`.
pub fn sync_results<T: jope_client::Transport>(
    local: &T,
    run: &jope_api::ids::RunId,
    expected: Option<&(String, String)>,
) -> Result<Vec<String>> {
    let snapshot = local.run_status(run)?;
    anyhow::ensure!(
        snapshot.state == jope_api::RunState::Completed,
        "only a completed run's results are synced"
    );
    let version = local.system_info()?.version;
    let mut lines = vec![sync_snapshot_for(&snapshot, &version, expected)?];
    lines.push(
        sync_events(local, &snapshot).unwrap_or_else(|error| format!("Run events: {error:#}")),
    );
    let mut items = Vec::new();
    if let Some(best) = &snapshot.best {
        let request = jope_api::ExportRequest {
            algorithm_id: best.algorithm_id.clone(),
            kind: jope_api::ExportKind::Model,
            languages: Vec::new(),
            require_all_languages: false,
            refit: jope_api::Refit::FitHoldout,
        };
        match export(local, run, &request, MAX_MODEL)
            .and_then(|bytes| pack_object(&unpack_object(&bytes)?))
        {
            Ok(bytes) => items.push((Kind::Model, bytes)),
            Err(error) => lines.push(format!(
                "Results: the model could not be exported ({error:#})."
            )),
        }
    }
    if let Some(best) = &snapshot.best {
        let request = jope_api::ExportRequest {
            algorithm_id: best.algorithm_id.clone(),
            kind: jope_api::ExportKind::Source,
            languages: vec![jope_api::Language::Python, jope_api::Language::Rust],
            require_all_languages: false,
            refit: jope_api::Refit::FitHoldout,
        };
        match export(local, run, &request, 64 << 20).and_then(|bytes| flatten_code(&bytes)) {
            Ok(bytes) => items.push((Kind::Code, bytes)),
            Err(error) => lines.push(format!("Results: no generated code to sync ({error:#}).")),
        }
    }
    let _lock = connection_lock()?;
    let c = connection()?;
    if let Some((origin, account)) = expected {
        anyhow::ensure!(
            &c.origin == origin && &c.account_id == account,
            "VEOX account changed; results were not synced"
        );
    }
    let id = run.to_string();
    binding(&c, &id)?;
    let record: SyncRecord = serde_json::from_slice(&read_private(&record_path(&c, &id)?)?)?;
    let cloud = record.cloud_id.context("run is not synced")?;
    anyhow::ensure!(uuid(&cloud), "invalid cloud run");
    let state = account_root(&c)?
        .join("runs")
        .join(format!("{id}.results.json"));
    let client = Client::connected(&c)?.with_timeout(Duration::from_secs(120))?;
    let model = items
        .iter()
        .find(|(kind, _)| *kind == Kind::Model)
        .map(|(_, bytes)| (jope_runtime::sha256_hex(bytes), bytes.len() as u64));
    lines.extend(push(&client, &cloud, items, &state, &std::thread::sleep));
    // The event follows the model it names (the app checks it against the stored digest).
    if let Some((hash, bytes)) =
        model.filter(|(hash, _)| delivered_model(&state).as_deref() == Some(hash.as_str()))
    {
        lines.push(
            super::telemetry::record_publication(&c, &client, &id, &cloud, &hash, bytes)
                .map(|note| note.trim().to_owned())
                .unwrap_or_else(|error| format!("Model event queued: {error:#}")),
        );
    }
    lines.push(format!(
        "Explore it in your workspace: {}/runs/{cloud}",
        c.origin
    ));
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn retry_backs_off_on_transient_failures_and_stops_on_permanent_ones() {
        let sleeps = RefCell::new(Vec::new());
        let sleep = |d: Duration| sleeps.borrow_mut().push(d.as_secs());
        let result = retry(3, &sleep, |n| {
            if n < 2 {
                Attempt::Retry(format!("HTTP 503 #{n}"))
            } else {
                Attempt::Done(n)
            }
        });
        assert_eq!(result, Ok(2));
        assert_eq!(*sleeps.borrow(), vec![1, 2]);
        sleeps.borrow_mut().clear();
        let calls = RefCell::new(0);
        let result: std::result::Result<(), _> = retry(3, &sleep, |_| {
            *calls.borrow_mut() += 1;
            Attempt::Fail("refused (HTTP 403)".into())
        });
        assert_eq!(result, Err("refused (HTTP 403)".into()));
        assert_eq!(*calls.borrow(), 1);
        assert!(sleeps.borrow().is_empty());
        let result: std::result::Result<(), _> =
            retry(3, &sleep, |n| Attempt::Retry(format!("down {n}")));
        assert_eq!(result, Err("down 2".into()));
        assert_eq!(
            *sleeps.borrow(),
            vec![1, 2],
            "no sleep after the last attempt"
        );
    }

    #[test]
    fn code_exports_are_flattened_to_unique_plain_files() {
        let mut zip = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
        let o = zip::write::SimpleFileOptions::default();
        for (name, body) in [
            ("source/python/model.py", "py"),
            ("source/rust/model.rs", "rs"),
            ("source/rust/README.md", "a"),
            ("source/python/README.md", "b"),
            ("source/.hidden", "x"),
            ("../escape.py", "x"),
        ] {
            zip.start_file(name, o).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        let flat = flatten_code(&zip.finish().unwrap().into_inner()).unwrap();
        let mut archive = zip::ZipArchive::new(io::Cursor::new(flat)).unwrap();
        let mut names: Vec<String> = (0..archive.len())
            .map(|n| archive.by_index(n).unwrap().name().unwrap().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "README.md",
                "model.py",
                "model.rs",
                "source_python_README.md"
            ]
        );
    }
}
