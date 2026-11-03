//! Application-owned HTTPS account connection. The local engine never receives credentials.
mod client_auth;
mod commands;
mod sync;

pub use commands::{auth_command, cloud_command, default_workspace, spawn_login};
pub use sync::{
    fetch_model, publish_best, publish_file, publish_model, run_entries, sync_events,
    sync_snapshot, sync_snapshot_for, Upload,
};
#[cfg(test)]
use sync::{pack_object, unpack_object};

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
    /// Publish explicitly: the fitted model (the default), or with `--report`/`--code` only the
    /// named PDF report and ZIP of generated code. Nothing uploads without this command.
    Publish {
        run: String,
        #[arg(long)]
        model: Option<PathBuf>,
        /// A PDF report for the run's Report tab (16 MiB at most).
        #[arg(long)]
        report: Option<PathBuf>,
        /// A ZIP of the run's generated code, flat files only (8 MiB at most).
        #[arg(long)]
        code: Option<PathBuf>,
    },
    /// Anonymous, de-identified usage telemetry for connected runs: `on`, `off`, or show it.
    /// Off unless you turn it on.
    Telemetry {
        setting: Option<String>,
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

#[cfg(test)]
mod tests;
