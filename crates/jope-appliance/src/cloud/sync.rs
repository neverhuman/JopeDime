use super::*;

pub use telemetry::sync_events;
use telemetry::record_publication;
pub(super) use telemetry::telemetry_command;
pub use upload::{publish_file, Upload};

mod telemetry;
mod upload;

#[derive(Serialize, Deserialize)]
pub(crate) struct SyncRecord {
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

mod entries;
mod models;
mod snapshot;

pub use entries::{publish_best, run_entries};
pub use models::{fetch_model, publish_model};
pub use snapshot::{sync_snapshot, sync_snapshot_for};
pub(crate) use snapshot::{pack_object, unpack_object};
