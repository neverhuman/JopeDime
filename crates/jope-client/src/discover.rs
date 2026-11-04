//! Engine selection and bounded detached startup (spec §3.1).
//!
//! Callers supply the canonical `jope_paths::jopedime_home()` and their executable's
//! actual engine capability. The client's `engine` transport feature is a separate concern.
use crate::{http::Credentials, transport::Response, Http, Request};
use serde::Deserialize;
use std::{
    ffi::OsString,
    fmt, fs, io,
    path::PathBuf,
    process::Child,
    thread,
    time::{Duration, Instant},
};

/// Maximum time local discovery spends attaching to or starting an engine.
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
/// Message returned when no explicit, existing, or startable engine is available.
pub const NO_ENGINE: &str = "no engine found; run `jopedime serve` or pass `--endpoint`";
const MAX_METADATA: u64 = 16 * 1024;
const POLL: Duration = Duration::from_millis(20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// A foreground command may start a detached local engine.
    Interactive,
    /// A command that must not start a detached engine; it may use an in-process engine.
    Batch,
}

/// Inputs that control explicit endpoint selection and local engine discovery.
#[derive(Clone)]
pub struct Options {
    /// Canonical JopeDime home used for daemon metadata and local coordination.
    pub home: PathBuf,
    /// Executable used to start a detached engine when needed.
    pub executable: PathBuf,
    /// Optional authoritative endpoint; an invalid explicit value is an error.
    pub endpoint: Option<String>,
    /// Whether discovery may start a detached process or select an in-process engine.
    pub mode: Mode,
    /// Whether `executable` includes the engine, normally `cfg!(feature = "engine")`
    /// in the appliance, not in this client library.
    pub engine_available: bool,
}
impl fmt::Debug for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Options")
            .field("home", &self.home)
            .field("executable", &self.executable)
            .field("explicit_endpoint", &self.endpoint.is_some())
            .field("mode", &self.mode)
            .field("engine_available", &self.engine_available)
            .finish()
    }
}

#[derive(Debug)]
pub enum Engine {
    /// Authoritative explicit selection. Authentication/remote TLS belongs to the caller.
    Endpoint(String),
    /// An authenticated client attached to or started beside this process.
    Background(Background),
    /// The caller opens its in-process engine; discovery starts no listener.
    InProcess,
}

#[derive(Clone, Debug)]
pub struct Background {
    /// Authenticated client for the attached or newly started local engine.
    pub client: Http,
    /// Engine endpoint read from its published metadata.
    pub url: String,
    /// Process ID reported by the engine metadata.
    pub pid: u32,
    /// Engine package version.
    pub version: String,
    /// JopeDime API version reported by the engine.
    pub api_version: String,
    /// Engine startup timestamp from its metadata.
    pub started_at: String,
}

/// Errors encountered while validating, attaching to, or starting a local engine.
#[derive(Debug)]
pub enum DiscoveryError {
    /// The explicitly selected endpoint is not a valid supported authority.
    InvalidEndpoint,
    /// Local daemon metadata is malformed or inconsistent.
    InvalidMetadata,
    /// Reading local engine metadata or coordination state failed.
    Io(io::Error),
    /// The configured engine process could not be started.
    Spawn(io::Error),
    /// No engine was available and local startup was not permitted.
    NoEngine,
    /// Another local engine owns the store or startup coordination slot.
    Busy {
        /// PID of the process currently holding the store, when it could be identified.
        pid: Option<u32>,
        /// Command recorded for the current store owner.
        command: String,
    },
    /// The engine did not become ready before the startup deadline.
    StartupTimeout {
        /// Exit status of the child process, if it exited before the deadline.
        child_exit: Option<i32>,
    },
}
/// Backward-compatible public name for discovery failures.
pub use DiscoveryError as Error;

impl DiscoveryError {
    /// Map this discovery failure to the stable CLI exit-code contract.
    pub fn exit_code(&self) -> u8 {
        use jope_api::error::ErrorCode;
        match self {
            Self::InvalidEndpoint | Self::InvalidMetadata => 2,
            Self::Io(_) => ErrorCode::IoFailure.exit_code().unwrap(),
            Self::NoEngine | Self::Spawn(_) | Self::StartupTimeout { .. } => {
                ErrorCode::EngineUnreachable.exit_code().unwrap()
            }
            Self::Busy { .. } => ErrorCode::EngineBusy.exit_code().unwrap(),
        }
    }
}
impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEndpoint => f.write_str("invalid engine endpoint; use an HTTP(S) authority"),
            Self::InvalidMetadata => f.write_str("invalid daemon metadata; check the engine with `jopedime serve --status`"),
            Self::Io(e) => write!(f, "engine discovery IO: {e}"),
            Self::Spawn(e) => write!(f, "cannot start the engine: {e}"),
            Self::NoEngine => f.write_str(NO_ENGINE),
            Self::Busy { pid, command } => write!(f, "ENGINE_BUSY: store owner PID {} command {command:?}; retry or use another JOPEDIME_HOME", pid.map_or("unknown".into(), |p|p.to_string())),
            Self::StartupTimeout { child_exit } => write!(f, "engine did not become ready within 5 s (child exit: {child_exit:?}); run `jopedime serve` for diagnostics"),
        }
    }
}
impl std::error::Error for DiscoveryError {}
impl From<io::Error> for DiscoveryError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
struct Record {
    url: String,
    token: String,
    pid: u32,
    version: String,
    api_version: String,
    started_at: String,
}

/// Select an explicit endpoint, attach to a healthy local engine, or start/use a local engine.
///
/// An explicit endpoint is authoritative: validation or connection failures never silently fall
/// back to another engine. Local startup is bounded by [`STARTUP_TIMEOUT`].
pub fn discover(options: &Options) -> Result<Engine, Error> {
    discover_with(
        options,
        std::env::var_os("JOPEDIME_ENDPOINT"),
        STARTUP_TIMEOUT,
        start_engine,
    )
}

fn discover_with(
    options: &Options,
    environment: Option<OsString>,
    timeout: Duration,
    start: impl FnOnce(&Options) -> io::Result<Child>,
) -> Result<Engine, Error> {
    let selected = options.endpoint.clone().or_else(|| {
        environment
            .as_ref()
            .and_then(|v| v.to_str())
            .map(str::to_owned)
    });
    if options.endpoint.is_none() && environment.is_some() && selected.is_none() {
        return Err(Error::InvalidEndpoint);
    }
    if let Some(url) = selected {
        let uri: ureq::http::Uri = url.parse().map_err(|_| Error::InvalidEndpoint)?;
        if !matches!(uri.scheme_str(), Some("http" | "https"))
            || uri.host().is_none()
            || uri.port_u16() == Some(0)
            || !matches!(uri.path(), "" | "/")
            || uri.query().is_some()
            || url.contains(['@', '#'])
        {
            return Err(Error::InvalidEndpoint);
        }
        return Ok(Engine::Endpoint(url));
    }
    let deadline = Instant::now() + timeout;
    if let Some(engine) = attached(&options.home, deadline)? {
        return Ok(Engine::Background(engine));
    }
    if !options.engine_available {
        return Err(Error::NoEngine);
    }
    if options.mode == Mode::Batch {
        return Ok(Engine::InProcess);
    }
    private_home(&options.home)?;
    let mut start = Some(start);
    let mut child = Reaper(None);
    let mut child_exit = None;
    let mut coordination = None;
    let result = loop {
        if Instant::now() >= deadline {
            break Err(match store_owner(&options.home)? {
                Some((pid, command)) => Error::Busy { pid, command },
                None => Error::StartupTimeout { child_exit },
            });
        }
        if let Some(engine) = attached(&options.home, deadline)? {
            break Ok(Engine::Background(engine));
        }
        if coordination.is_none() && start.is_some() {
            let file = private_options()
                .write(true)
                .create(true)
                .open(options.home.join(".discover.lock"))?;
            private_file(&file)?;
            match file.try_lock() {
                Ok(()) => {
                    coordination = Some(file);
                    // Another client may have published immediately before we acquired this gate.
                    if let Some(engine) = attached(&options.home, deadline)? {
                        break Ok(Engine::Background(engine));
                    }
                    child.0 = Some(start.take().unwrap()(options).map_err(Error::Spawn)?);
                }
                Err(fs::TryLockError::WouldBlock) => {}
                Err(fs::TryLockError::Error(e)) => break Err(Error::Io(e)),
            }
        }
        if let Some(process) = &mut child.0 {
            if let Some(status) = process.try_wait()? {
                child_exit = status.code();
                child.0 = None;
                // Includes a losing store-lock race, regardless of the child's exit code.
                // Re-read the shared descriptor until the same original deadline.
            }
        }
        thread::sleep(POLL.min(deadline.saturating_duration_since(Instant::now())));
    };
    drop(coordination);
    result
}

mod local;
#[cfg(test)]
use local::{alive, detached, record};
use local::{
    attached, private_file, private_home, private_options, start_engine, store_owner, Reaper,
};

#[cfg(test)]
mod tests;
