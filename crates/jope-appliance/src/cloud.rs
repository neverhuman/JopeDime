//! Application-owned HTTPS account connection. The local engine never receives credentials.
use anyhow::{Context, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const MAX_JSON: u64 = 2 * 1024 * 1024;
pub const MAX_MODEL: u64 = 32 * 1024 * 1024;
#[derive(Subcommand)]
pub enum AuthCommand {
    Login {
        #[arg(long)]
        workspace: Option<String>,
        #[arg(long, conflicts_with = "token_stdin")]
        token_file: Option<PathBuf>,
        #[arg(long)]
        token_stdin: bool,
        #[arg(long)]
        gateway_auth_file: Option<PathBuf>,
        #[arg(long)]
        no_browser: bool,
    },
    Logout,
    Status,
}
#[derive(Subcommand)]
pub enum CloudCommand {
    Sync {
        #[arg(long)]
        run: Option<String>,
    },
    Runs,
    Publish {
        run: String,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    Fetch {
        run: String,
    },
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Gateway {
    pub origin: String,
    pub username: String,
    pub password: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Connection {
    pub origin: String,
    pub account_id: String,
    pub email: String,
    token: String,
    #[serde(default)]
    gateway: Option<Gateway>,
    pub may_publish: bool,
}
#[derive(Clone)]
pub struct Client {
    pub origin: String,
    token: Option<String>,
    gateway: Option<Gateway>,
    agent: ureq::Agent,
}
#[derive(Deserialize)]
pub struct Device {
    #[serde(skip)]
    generation: String,
    device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: String,
    pub expires_in: u64,
    pub interval: u64,
}
pub fn origin(value: &str) -> Result<String> {
    let uri: ureq::http::Uri = value.parse().context("invalid workspace origin")?;
    let scheme = uri.scheme_str().unwrap_or("");
    anyhow::ensure!(
        scheme == "https"
            || (scheme == "http" && matches!(uri.host(), Some("127.0.0.1" | "localhost"))),
        "workspace must use HTTPS"
    );
    anyhow::ensure!(
        uri.host().is_some()
            && matches!(uri.path(), "" | "/")
            && uri.query().is_none()
            && !value.contains(['@', '#', '\\'])
            && !value.chars().any(char::is_control),
        "workspace must be an origin without credentials, path, query or fragment"
    );
    Ok(format!(
        "{scheme}://{}",
        uri.authority().context("workspace has no host")?
    )
    .to_ascii_lowercase())
}
fn cloud_root() -> Result<PathBuf> {
    let path = jope_paths::jopedime_home().join("cloud");
    private_dir(&path)?;
    Ok(path)
}
fn private_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                private_dir(parent)?;
            }
        }
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(path)?;
    }
    let meta = fs::symlink_metadata(path)?;
    anyhow::ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "cloud directory must not be a symlink"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            meta.permissions().mode() & 0o077 == 0,
            "cloud directory must be private (0700)"
        );
    }
    Ok(())
}
fn read_private(path: &Path) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    anyhow::ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "credential/state file must be a regular file, not a symlink"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            meta.permissions().mode() & 0o077 == 0,
            "credential/state file must be private (0600)"
        );
    }
    anyhow::ensure!(meta.len() <= MAX_JSON, "credential/state file is too large");
    Ok(fs::read(path)?)
}
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("state file has no parent")?;
    private_dir(parent)?;
    if path.exists() || path.is_symlink() {
        read_private(path)?;
    }
    let temp = parent.join(format!(".write-{}", ulid::Ulid::new()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| -> Result<()> {
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
pub fn connection() -> Result<Connection> {
    let data = read_private(&cloud_root()?.join("connection.json"))?;
    let connection: Connection =
        serde_json::from_slice(&data).context("invalid connection file")?;
    anyhow::ensure!(
        origin(&connection.origin)? == connection.origin,
        "invalid stored workspace"
    );
    Ok(connection)
}
pub fn account_label() -> String {
    match connection() {
        Ok(c) => format!("VEOX: {}", c.email),
        Err(_) => "Cloud not connected · /login".into(),
    }
}
impl Client {
    pub fn new(workspace: &str, token: Option<String>, gateway: Option<Gateway>) -> Result<Self> {
        let tls = ureq::tls::TlsConfig::builder()
            .unversioned_rustls_crypto_provider(Arc::new(
                rustls::crypto::aws_lc_rs::default_provider(),
            ))
            .build();
        let config = ureq::Agent::config_builder()
            .tls_config(tls)
            .proxy(None)
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(20)))
            .build();
        let workspace = origin(workspace)?;
        if let Some(gateway) = &gateway {
            anyhow::ensure!(
                origin(&gateway.origin)? == workspace,
                "gateway credential file belongs to a different workspace"
            );
        }
        Ok(Self {
            origin: workspace,
            token,
            gateway,
            agent: ureq::Agent::new_with_config(config),
        })
    }
    pub fn connected(c: &Connection) -> Result<Self> {
        Self::new(&c.origin, Some(c.token.clone()), c.gateway.clone())
    }
    fn wire(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<ureq::http::Request<Vec<u8>>> {
        anyhow::ensure!(
            path.starts_with("/api/") && !path.contains(['#', '\r', '\n']),
            "invalid workspace API path"
        );
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.origin))
            .header("content-type", content_type)
            .body(body)?;
        if let Some(gateway) = &self.gateway {
            let basic = basic_auth(&gateway.username, &gateway.password);
            let mut value = ureq::http::HeaderValue::from_str(&basic)?;
            value.set_sensitive(true);
            request.headers_mut().insert("authorization", value);
        }
        if let Some(token) = &self.token {
            let (name, value) = if self.gateway.is_some() {
                ("x-veox-api-token", token.clone())
            } else {
                ("authorization", format!("Bearer {token}"))
            };
            let mut value = ureq::http::HeaderValue::from_str(&value)?;
            value.set_sensitive(true);
            request.headers_mut().insert(name, value);
        }
        Ok(request)
    }
    fn call(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
        content_type: &str,
        limit: u64,
    ) -> Result<(u16, Vec<u8>)> {
        let request = self.wire(method, path, body, content_type)?;
        let mut response = self
            .agent
            .run(request)
            .context("workspace request failed")?;
        let status = response.status().as_u16();
        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .take(limit + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 <= limit,
            "workspace response exceeded its size limit"
        );
        Ok((status, bytes))
    }
    pub fn json(&self, method: &str, path: &str, body: Value) -> Result<Value> {
        let (status, bytes) = self.call(
            method,
            path,
            if matches!(method, "GET" | "DELETE") {
                Vec::new()
            } else {
                serde_json::to_vec(&body)?
            },
            "application/json",
            MAX_JSON,
        )?;
        anyhow::ensure!(
            (200..300).contains(&status),
            "workspace request refused (HTTP {status}); check login, permissions and connection"
        );
        serde_json::from_slice(&bytes).context("workspace returned invalid JSON")
    }
    fn form(&self, path: &str, values: &[(&str, &str)]) -> Result<(u16, Value)> {
        let body = values
            .iter()
            .map(|(key, value)| format!("{}={}", encode(key), encode(value)))
            .collect::<Vec<_>>()
            .join("&");
        let (status, bytes) = self.call(
            "POST",
            path,
            body.into_bytes(),
            "application/x-www-form-urlencoded",
            MAX_JSON,
        )?;
        Ok((
            status,
            serde_json::from_slice(&bytes).context("workspace returned invalid login JSON")?,
        ))
    }
    pub fn start_login(&self) -> Result<Device> {
        let (status, value) = self.form(
            "/api/cli/device",
            &[
                ("client_id", "jain-cli"),
                ("scope", "cli"),
                ("client_name", "jAIn CLI"),
            ],
        )?;
        anyhow::ensure!(
            status == 200,
            "workspace could not start login (HTTP {status})"
        );
        let mut device: Device = serde_json::from_value(value)?;
        anyhow::ensure!(
            device.verification_uri == format!("{}/cli/connect", self.origin)
                && device.verification_uri_complete
                    == format!("{}?user_code={}", device.verification_uri, device.user_code)
                && device
                    .user_code
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "workspace returned an invalid login URL"
        );
        anyhow::ensure!(
            device.expires_in <= 900 && (1..=60).contains(&device.interval),
            "invalid login lifetime or polling interval"
        );
        device.generation = ulid::Ulid::new().to_string();
        let _lock = connection_lock()?;
        write_private(
            &cloud_root()?.join("auth-generation"),
            device.generation.as_bytes(),
        )?;
        write_private(
            &cloud_root()?.join("workspace-origin"),
            self.origin.as_bytes(),
        )?;
        Ok(device)
    }
    pub fn finish_login(&self, device: &Device, cancel: &AtomicBool) -> Result<Connection> {
        let start = Instant::now();
        let mut interval = device.interval;
        loop {
            for _ in 0..interval * 10 {
                anyhow::ensure!(!cancel.load(Ordering::SeqCst), "login cancelled");
                std::thread::sleep(Duration::from_millis(100));
            }
            anyhow::ensure!(
                start.elapsed().as_secs() < device.expires_in,
                "login code expired; start /login again"
            );
            let (status, value) = self.form(
                "/api/cli/token",
                &[
                    ("client_id", "jain-cli"),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", &device.device_code),
                ],
            )?;
            if status == 200 {
                let token = value["access_token"]
                    .as_str()
                    .context("login returned no credential")?
                    .to_owned();
                return self.save_token_checked(token, Some((&device.generation, cancel)));
            }
            match value["error"].as_str() {
                Some("authorization_pending") => {}
                Some("slow_down") => interval = (interval + 5).min(60),
                Some("access_denied") => anyhow::bail!("login was denied; no account connected"),
                Some("expired_token") => anyhow::bail!("login code expired; start /login again"),
                _ => anyhow::bail!("workspace refused login (HTTP {status})"),
            }
        }
    }
    pub fn save_token(&self, token: String) -> Result<Connection> {
        self.save_token_checked(token, None)
    }
    fn save_token_checked(
        &self,
        token: String,
        pending: Option<(&str, &AtomicBool)>,
    ) -> Result<Connection> {
        anyhow::ensure!(
            !token.is_empty() && token.len() <= 512 && !token.chars().any(char::is_control),
            "invalid account token"
        );
        let client = Self::new(&self.origin, Some(token.clone()), self.gateway.clone())?;
        let account = client.json("GET", "/api/cli/account", Value::Null)?;
        let c = Connection {
            origin: self.origin.clone(),
            account_id: account["accountId"]
                .as_str()
                .context("workspace returned no account identity")?
                .into(),
            email: account["email"]
                .as_str()
                .context("workspace returned no account address")?
                .into(),
            token,
            gateway: self.gateway.clone(),
            may_publish: account["mayPublish"].as_bool().unwrap_or(false),
        };
        let _lock = connection_lock()?;
        if let Some((generation, cancel)) = pending {
            let current = read_private(&cloud_root()?.join("auth-generation"))?;
            if cancel.load(Ordering::SeqCst) || current != generation.as_bytes() {
                let _ = client.json("DELETE", "/api/cli/connection", Value::Null);
                anyhow::bail!("login cancelled or replaced by another login");
            }
        } else {
            write_private(
                &cloud_root()?.join("auth-generation"),
                ulid::Ulid::new().to_string().as_bytes(),
            )?;
        }
        write_private(
            &cloud_root()?.join("connection.json"),
            &serde_json::to_vec(&c)?,
        )?;
        write_private(
            &cloud_root()?.join("workspace-origin"),
            self.origin.as_bytes(),
        )?;
        Ok(c)
    }
}
fn encode(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}
fn basic_auth(username: &str, password: &str) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let value = format!("{username}:{password}");
    let mut encoded = String::new();
    for chunk in value.as_bytes().chunks(3) {
        let a = chunk[0] as usize;
        let b = *chunk.get(1).unwrap_or(&0) as usize;
        let c = *chunk.get(2).unwrap_or(&0) as usize;
        encoded.push(TABLE[a >> 2] as char);
        encoded.push(TABLE[((a & 3) << 4) | (b >> 4)] as char);
        encoded.push(if chunk.len() > 1 {
            TABLE[((b & 15) << 2) | (c >> 6)] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            TABLE[c & 63] as char
        } else {
            '='
        });
    }
    format!("Basic {encoded}")
}
pub fn gateway(path: Option<&Path>, workspace: &str) -> Result<Option<Gateway>> {
    let workspace = origin(workspace)?;
    let path = path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("JAIN_GATEWAY_AUTH_FILE").map(PathBuf::from))
        .or_else(|| {
            cloud_root()
                .ok()
                .map(|root| {
                    root.join("gateways").join(format!(
                        "{}.json",
                        jope_runtime::sha256_hex(workspace.as_bytes())
                    ))
                })
                .filter(|path| path.exists())
        });
    path.map(|path| {
        let g: Gateway = serde_json::from_slice(&read_private(&path)?)
            .context("invalid gateway credential file")?;
        anyhow::ensure!(
            !g.username.contains(':') && !g.username.chars().any(char::is_control),
            "invalid gateway username"
        );
        Ok(g)
    })
    .transpose()
}
pub fn open_browser(url: &str) -> Result<()> {
    let uri: ureq::http::Uri = url.parse()?;
    anyhow::ensure!(
        uri.scheme_str() == Some("https") || matches!(uri.host(), Some("localhost" | "127.0.0.1")),
        "invalid browser URL"
    );
    #[cfg(target_os = "macos")]
    let command = "open";
    #[cfg(not(target_os = "macos"))]
    let command = "xdg-open";
    std::process::Command::new(command)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    Ok(())
}
pub fn logout() -> Result<String> {
    let _lock = connection_lock()?;
    write_private(
        &cloud_root()?.join("auth-generation"),
        ulid::Ulid::new().to_string().as_bytes(),
    )?;
    let path = cloud_root()?.join("connection.json");
    if !path.exists() && !path.is_symlink() {
        return Ok("Not signed in; pending login cancelled.".into());
    }
    let c = connection()?;
    let revoked = Client::connected(&c)?
        .json("DELETE", "/api/cli/connection", Value::Null)
        .is_ok();
    fs::remove_file(cloud_root()?.join("connection.json"))?;
    Ok(if revoked {
        "Signed out; this CLI credential was revoked.".into()
    } else {
        "Signed out locally. Remote revocation could not be confirmed; revoke the token in your VEOX account.".into()
    })
}
pub fn auth_command(command: AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Status => println!("{}", account_label()),
        AuthCommand::Logout => println!("{}", logout()?),
        AuthCommand::Login {
            workspace,
            token_file,
            token_stdin,
            gateway_auth_file,
            no_browser,
        } => {
            let workspace = workspace.unwrap_or_else(default_workspace);
            let client = Client::new(
                &workspace,
                None,
                gateway(gateway_auth_file.as_deref(), &workspace)?,
            )?;
            let c = if token_file.is_some() || token_stdin {
                let mut token = String::new();
                if let Some(path) = token_file {
                    token = String::from_utf8(read_private(&path)?)?;
                } else {
                    anyhow::ensure!(!io::stdin().is_terminal(),"use browser login interactively; API tokens require a private file or redirected stdin");
                    io::stdin().take(513).read_to_string(&mut token)?;
                }
                client.save_token(token.trim().to_owned())?
            } else {
                let device = client.start_login()?;
                println!(
                    "Open {}\nPairing code: {}\nSign in and approve the CLI in your browser.",
                    device.verification_uri_complete, device.user_code
                );
                io::stdout().flush()?;
                if !no_browser {
                    let _ = open_browser(&device.verification_uri_complete);
                }
                client.finish_login(&device, &AtomicBool::new(false))?
            };
            println!("Signed in to {} as {}", c.origin, c.email);
        }
    }
    Ok(())
}
pub fn cloud_command(command: CloudCommand) -> Result<()> {
    use jope_client::Transport;
    match command {
        CloudCommand::Runs => {
            let (entries, notice) = run_entries(&local_client()?)?;
            println!("{notice}");
            for row in entries {
                println!(
                    "{} · {} · {}{}",
                    row.id,
                    row.status,
                    row.availability(),
                    row.cloud_url
                        .map(|url| format!(" · {url}"))
                        .unwrap_or_default()
                );
            }
        }
        CloudCommand::Sync { run } => {
            let local = local_client()?;
            let version = local.system_info()?.version;
            let runs = if let Some(run) = run {
                vec![local.run_status(&run.parse()?)?]
            } else {
                crate::terminal::runs::local_runs(&local)?
            };
            for snapshot in runs {
                println!("{}", sync_snapshot(&snapshot, &version)?);
            }
        }
        CloudCommand::Publish { run, model } => println!(
            "{}",
            if let Some(model) = model {
                publish_model(&run, &model)?
            } else {
                publish_best(&run)?
            }
        ),
        CloudCommand::Fetch { run } => println!("{}", fetch_model(&run)?),
    }
    Ok(())
}

pub fn spawn_login(
    workspace: String,
    sender: std::sync::mpsc::Sender<Vec<String>>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = (|| -> Result<String> {
            let client = Client::new(&workspace, None, gateway(None, &workspace)?)?;
            let device = client.start_login()?;
            let _ = sender.send(vec![
                format!("Open {}", device.verification_uri_complete),
                format!(
                    "Pairing code: {} · sign in and approve this CLI in your browser.",
                    device.user_code
                ),
            ]);
            let _ = open_browser(&device.verification_uri_complete);
            let c = client.finish_login(&device, &cancel)?;

            Ok(format!(
                "Signed in to {} as {}. Local datasets stay local; model publication is explicit.",
                c.origin, c.email
            ))
        })();
        let _ = sender.send(vec![
            result.unwrap_or_else(|error| format!("Login: {error:#}"))
        ]);
    });
}
pub fn default_workspace() -> String {
    std::env::var("JAIN_WORKSPACE")
        .ok()
        .or_else(|| connection().ok().map(|c| c.origin))
        .or_else(|| {
            cloud_root()
                .ok()
                .and_then(|root| read_private(&root.join("workspace-origin")).ok())
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .and_then(|value| origin(&value).ok())
        })
        .unwrap_or_else(|| "https://app.veox.ai".into())
}

fn connection_lock() -> Result<fs::File> {
    let path = cloud_root()?.join("connection.lock");
    if path.exists() || path.is_symlink() {
        read_private(&path)?;
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    file.lock()?;
    Ok(file)
}

#[derive(Serialize, Deserialize)]
struct SyncRecord {
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
fn pack_object(object: &jope_runtime::model_object::ModelObject) -> Result<Vec<u8>> {
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
fn unpack_object(bytes: &[u8]) -> Result<jope_runtime::model_object::ModelObject> {
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
    Ok(format!(
        "Published model {} · SHA-256 {hash}\n{}/runs/{cloud}",
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_origin_rejects_credential_leaks_and_plain_remote_http() {
        for url in [
            "http://evil.example",
            "https://u:p@example.test",
            "https://example.test/path",
            "https://example.test?token=secret",
            "https://example.test/#x",
        ] {
            assert!(origin(url).is_err(), "{url}");
        }
        assert_eq!(
            origin("https://APP.veox.ai/").unwrap(),
            "https://app.veox.ai"
        );
    }
    /// Non-secret test fixtures: these values only exercise header construction and never reach a server.
    const FIXTURE_ACCOUNT_TOKEN: &str = "fixture-account-token";
    const FIXTURE_GATEWAY_USER: &str = "dev";
    const FIXTURE_GATEWAY_PASS: &str = "fixture-gateway-pass";

    #[test]
    fn basic_gateway_and_account_credentials_are_distinct_sensitive_headers() {
        let c = Client::new(
            "https://example.test",
            Some(FIXTURE_ACCOUNT_TOKEN.into()),
            Some(Gateway {
                origin: "https://example.test".into(),
                username: FIXTURE_GATEWAY_USER.into(),
                password: FIXTURE_GATEWAY_PASS.into(),
            }),
        )
        .unwrap();
        let request = c
            .wire("GET", "/api/cli/account", Vec::new(), "application/json")
            .unwrap();
        assert!(request.headers()["authorization"].is_sensitive());
        assert!(request.headers()["x-veox-api-token"].is_sensitive());
        assert_ne!(
            request.headers()["authorization"],
            request.headers()["x-veox-api-token"]
        );
        assert_eq!(basic_auth("u", "p"), "Basic dTpw");
    }

    #[test]
    fn private_files_refuse_symlinks_and_group_readable_credentials() {
        let root = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = root.path().join("credential.json");
        write_private(&path, b"synthetic credential").unwrap();
        assert_eq!(read_private(&path).unwrap(), b"synthetic credential");
        #[cfg(unix)]
        {
            use std::os::unix::fs::{symlink, PermissionsExt};
            let link = root.path().join("link");
            symlink(&path, &link).unwrap();
            assert!(read_private(&link).is_err());
            assert!(write_private(&link, b"replacement").is_err());
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
            assert!(read_private(&path).is_err());
        }
    }
    #[test]
    fn model_transfer_preserves_exact_identity_and_refuses_changed_parts() {
        use jope_runtime::model_object::{Manifest, ModelObject, Part, Producer, SCHEMA};
        let data = b"opaque fixture".to_vec();
        let manifest = Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: "org.example/opaque/v1".into(),
            producer: Producer::default(),
            primary: "state.bin".into(),
            required_capabilities: vec!["org.example/opaque/v1".into()],
            files: vec![Part::new("state.bin", "application/octet-stream", &data)],
            metadata: json!({}),
        };
        let object = ModelObject::new(
            manifest,
            std::collections::BTreeMap::from([("state.bin".into(), data)]),
            MAX_MODEL,
        )
        .unwrap();
        let bytes = pack_object(&object).unwrap();
        let decoded = unpack_object(&bytes).unwrap();
        assert_eq!(object.id(), decoded.id());
        assert_eq!(object.manifest_bytes(), decoded.manifest_bytes());
        let mut zip = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("manifest.json", options).unwrap();
        zip.write_all(object.manifest_bytes()).unwrap();
        zip.start_file("state.bin", options).unwrap();
        zip.write_all(b"changed fixture").unwrap();
        assert!(unpack_object(&zip.finish().unwrap().into_inner()).is_err());
    }
    #[test]
    fn gateway_credentials_cannot_be_forwarded_to_another_origin() {
        assert!(Client::new(
            "https://app.veox.ai",
            None,
            Some(Gateway {
                origin: "https://dev.example.test".into(),
                username: "dev".into(),
                password: "synthetic".into()
            })
        )
        .is_err());
    }
}
