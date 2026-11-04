use super::*;

// ---- connected-run events: veox.connected-run-events/v1 to the account, de-identified to
// veox-telemetry. See `cloud_events`.

impl Client {
    /// `POST /v1/telemetry` on a telemetry origin. No credential of any kind is attached.
    fn post_telemetry(&self, body: Vec<u8>) -> Result<u16> {
        let request = ureq::http::Request::builder()
            .method("POST")
            .uri(format!("{}/v1/telemetry", self.origin))
            .header("content-type", "application/json")
            .body(body)?;
        let mut response = self
            .agent
            .run(request)
            .context("telemetry request failed")?;
        let mut sink = Vec::new();
        let _ = response
            .body_mut()
            .as_reader()
            .take(64 * 1024)
            .read_to_end(&mut sink);
        Ok(response.status().as_u16())
    }
}

const TELEMETRY_ORIGIN: &str = "https://telemetry.veox.ai";

fn random_hex() -> String {
    format!("{:032x}", rand::random::<u128>())
}

fn telemetry_enabled() -> bool {
    cloud_root()
        .ok()
        .and_then(|root| read_private(&root.join("telemetry")).ok())
        .is_some_and(|bytes| bytes == b"basic")
}

pub(in crate::cloud) fn telemetry_command(setting: Option<String>) -> Result<String> {
    let path = cloud_root()?.join("telemetry");
    match setting.as_deref() {
        Some("on") => write_private(&path, b"basic")?,
        Some("off") => write_private(&path, b"off")?,
        Some(_) => anyhow::bail!("use `on` or `off`"),
        None => {}
    }
    Ok(if telemetry_enabled() {
        "Telemetry on: connected runs also send de-identified counts (bucketed sizes, scores in tenths, no names, no rows) to telemetry.veox.ai.".into()
    } else {
        "Telemetry off. Your account still receives the run details you sync; `jopedime cloud telemetry on` also sends de-identified counts to telemetry.veox.ai.".into()
    })
}

fn install_id() -> Result<String> {
    let path = cloud_root()?.join("telemetry-install-id");
    if path.exists() {
        let value = String::from_utf8(read_private(&path)?)?;
        if value.len() == 32 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Ok(value);
        }
    }
    let value = random_hex();
    write_private(&path, value.as_bytes())?;
    Ok(value)
}

fn session_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(random_hex)
}

/// What has been delivered for one run, per destination, and its telemetry pseudonym.
#[derive(Serialize, Deserialize, Default)]
struct EventState {
    sent_app: std::collections::BTreeSet<String>,
    sent_telemetry: std::collections::BTreeSet<String>,
    telemetry_run: Option<String>,
}

fn event_state_path(c: &Connection, run: &str) -> Result<PathBuf> {
    Ok(account_root(c)?
        .join("runs")
        .join(format!("{run}.events.json")))
}

fn read_event_state(path: &Path) -> Result<EventState> {
    if path.exists() {
        Ok(serde_json::from_slice(&read_private(path)?)?)
    } else {
        Ok(EventState::default())
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Send `events` the account has not yet acknowledged; then, when telemetry is on, their
/// de-identified forms. Caller holds the connection lock.
fn deliver(
    c: &Connection,
    client: &Client,
    run: &str,
    cloud: &str,
    events: &[Value],
    extra_telemetry: Option<(String, u64)>,
) -> Result<String> {
    let path = event_state_path(c, run)?;
    let mut state = read_event_state(&path)?;
    let pending: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["id"]
                .as_str()
                .is_some_and(|id| !state.sent_app.contains(id))
        })
        .collect();
    let mut sent = 0;
    for chunk in pending.chunks(crate::cloud_events::APP_BATCH) {
        let body = json!({"schema": crate::cloud_events::SCHEMA, "events": chunk});
        let (status, _) = client.call(
            "POST",
            &format!("/api/cli/runs/{cloud}/events"),
            serde_json::to_vec(&body)?,
            "application/json",
            MAX_JSON,
        )?;
        anyhow::ensure!(
            status == 200,
            "run events are queued locally; cloud refused them (HTTP {status})"
        );
        for event in chunk {
            if let Some(id) = event["id"].as_str() {
                state.sent_app.insert(id.to_owned());
            }
        }
        sent += chunk.len();
        write_private(&path, &serde_json::to_vec(&state)?)?;
    }
    let mut note = format!("Run events: {sent} sent");
    if telemetry_enabled() {
        let run_pseudonym = state.telemetry_run.get_or_insert_with(random_hex).clone();
        write_private(&path, &serde_json::to_vec(&state)?)?;
        let install = install_id()?;
        let ids = crate::cloud_events::Pseudonyms {
            install_id: &install,
            session_id: session_id(),
            run_id: &run_pseudonym,
            environment: crate::cloud_events::environment(&c.origin),
        };
        let mut envelopes: Vec<(String, Value)> =
            crate::cloud_events::telemetry(events, &ids, now_ms(), random_hex)
                .into_iter()
                .filter(|(id, _)| !state.sent_telemetry.contains(id))
                .collect();
        if let Some((id, bytes)) = extra_telemetry {
            if !state.sent_telemetry.contains(&id) {
                let envelope =
                    crate::cloud_events::telemetry_model(&ids, bytes, now_ms(), random_hex());
                envelopes.push((id, envelope));
            }
        }
        if !envelopes.is_empty() {
            let origin =
                std::env::var("JOPE_TELEMETRY_ORIGIN").unwrap_or_else(|_| TELEMETRY_ORIGIN.into());
            let telemetry = Client::new(&origin, None, None)?;
            for chunk in envelopes.chunks(crate::cloud_events::TELEMETRY_BATCH) {
                let body: Vec<&Value> = chunk.iter().map(|(_, v)| v).collect();
                let status = telemetry.post_telemetry(serde_json::to_vec(&body)?)?;
                anyhow::ensure!(
                    (200..300).contains(&status),
                    "telemetry refused (HTTP {status}); it is retried on the next sync"
                );
                for (id, _) in chunk {
                    state.sent_telemetry.insert(id.clone());
                }
                write_private(&path, &serde_json::to_vec(&state)?)?;
            }
            note.push_str(" · de-identified telemetry sent");
        }
    }
    Ok(note)
}

/// Sync a run's leaderboard, start, finish and dataset column profile to its connected record.
/// The run's metadata must already be synced (it gives the cloud run id).
pub fn sync_events<T: jope_client::Transport>(
    local: &T,
    snapshot: &jope_api::RunSnapshot,
) -> Result<String> {
    let events = crate::cloud_events::collect(local, snapshot)?;
    let _lock = connection_lock()?;
    let c = connection()?;
    anyhow::ensure!(
        c.may_publish,
        "this token is read-only; reconnect with CLI publication access"
    );
    let run = snapshot.run_id.to_string();
    binding(&c, &run)?;
    let record: SyncRecord = serde_json::from_slice(&read_private(&record_path(&c, &run)?)?)
        .context("sync the run before its events")?;
    let cloud = record.cloud_id.context("run is not synced")?;
    anyhow::ensure!(uuid(&cloud), "invalid cloud run");
    deliver(&c, &Client::connected(&c)?, &run, &cloud, &events, None)
}

/// Record an explicit model publication as an event (and, with telemetry on, its size bucket).
/// Caller holds the connection lock.
pub(super) fn record_publication(
    c: &Connection,
    client: &Client,
    run: &str,
    cloud: &str,
    hash: &str,
    bytes: u64,
) -> Result<String> {
    let at = chrono_like_now();
    let event = crate::cloud_events::model_published(hash, &at, now_ms() / 1000);
    let id = event["id"].as_str().unwrap_or_default().to_owned();
    deliver(c, client, run, cloud, &[event], Some((id, bytes))).map(|n| format!("\n{n}"))
}

/// RFC 3339 UTC, seconds precision, without a date-time dependency.
fn chrono_like_now() -> String {
    let secs = now_ms() / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil-from-days (Howard Hinnant), valid for the Unix era.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
