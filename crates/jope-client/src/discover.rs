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
pub const NO_ENGINE: &str = "no engine found; run `jain serve` or pass `--endpoint`";
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
#[allow(clippy::large_enum_variant)] // One value per discovery; boxing would change the public API.
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
    /// The source commit the engine was built from, when its build declared one. Release
    /// candidates share one X.Y.Z, so this (not the version) tells two builds apart (#2141).
    pub build: Option<String>,
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
    /// The running engine is another jAIn version (or an executable since replaced) and is busy
    /// with a run, so it was not restarted underneath that run.
    OutdatedBusy {
        /// The engine's version.
        version: String,
        /// The runs it is executing.
        running: Vec<String>,
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
            Self::Busy { .. } | Self::OutdatedBusy { .. } => {
                ErrorCode::EngineBusy.exit_code().unwrap()
            }
        }
    }
}
impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEndpoint => f.write_str("invalid engine endpoint; use an HTTP(S) authority"),
            Self::InvalidMetadata => f.write_str("invalid daemon metadata; check the engine with `jain serve --status`"),
            Self::Io(e) => write!(f, "engine discovery IO: {e}"),
            Self::Spawn(e) => write!(f, "cannot start the engine: {e}"),
            Self::NoEngine => f.write_str(NO_ENGINE),
            Self::Busy { pid, command } => write!(f, "ENGINE_BUSY: store owner PID {} command {command:?}; retry or use another JOPEDIME_HOME", pid.map_or("unknown".into(), |p|p.to_string())),
            Self::OutdatedBusy { version, running } => write!(f, "An older jAIn engine ({version}) is still running a job ({}). Wait for it or run `jain serve --stop --force`.", running.join(", ")),
            Self::StartupTimeout { child_exit } => write!(f, "engine did not become ready within 5 s (child exit: {child_exit:?}); run `jain serve` for diagnostics"),
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
    /// Absent in descriptors written by engines before 1.0.0-rc.3.
    #[serde(default)]
    build: Option<String>,
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

/// The client's own version: every engine it attaches to must match it.
pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The source commit this client was built from (`JOPE_BUILD_SOURCE_SHA`, 40 hex digits), or
/// `None` for a development build that declares none. The engine records the same value in its
/// descriptor, so an engine from another build of the same version is recognised as outdated.
pub fn client_build() -> Option<&'static str> {
    valid_build(option_env!("JOPE_BUILD_SOURCE_SHA"))
}

/// A declared build identity: exactly 40 hexadecimal digits.
pub fn valid_build(build: Option<&str>) -> Option<&str> {
    build.filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// What an attached engine must be: this client's version and, when known, its build.
#[derive(Clone, Copy, Debug)]
struct Expected<'a> {
    version: &'a str,
    build: Option<&'a str>,
}

impl Expected<'_> {
    fn label(&self) -> String {
        label(self.version, self.build)
    }
}

/// `1.0.0`, or `1.0.0 (build 51c0c1a)` when a build is known.
fn label(version: &str, build: Option<&str>) -> String {
    match build {
        Some(build) => format!("{version} (build {})", &build[..build.len().min(7)]),
        None => version.to_owned(),
    }
}

static NOTICE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// A one-line notice from the last discovery (an outdated engine was restarted), taken once.
pub fn take_notice() -> Option<String> {
    NOTICE.lock().ok()?.take()
}

fn set_notice(line: String) {
    if let Ok(mut notice) = NOTICE.lock() {
        *notice = Some(line);
    }
}

fn discover_with(
    options: &Options,
    environment: Option<OsString>,
    timeout: Duration,
    start: impl FnOnce(&Options) -> io::Result<Child>,
) -> Result<Engine, Error> {
    discover_versioned(
        options,
        environment,
        timeout,
        start,
        Some(Expected {
            version: CLIENT_VERSION,
            build: client_build(),
        }),
    )
}

/// [`discover_with`] with the version (and build) an attached engine must report (`None`: any).
fn discover_versioned(
    options: &Options,
    environment: Option<OsString>,
    timeout: Duration,
    start: impl FnOnce(&Options) -> io::Result<Child>,
    expected: Option<Expected<'_>>,
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
        if let Some(engine) = reconcile(&options.home, engine, expected)? {
            return Ok(Engine::Background(engine));
        }
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

/// Why an attached engine is not this client's, if it is not: another version, another build
/// of the same version (release candidates all report X.Y.Z; #2141), or a replaced executable.
fn outdated(engine: &Background, expected: Option<Expected<'_>>) -> Option<String> {
    let expected = expected?;
    if engine.version != expected.version {
        return Some(engine.version.clone());
    }
    if let Some(build) = expected.build {
        // An engine that declares no build predates build identities (or is a development
        // build): it cannot be shown to be this one, so it is treated as another build.
        let theirs = valid_build(engine.build.as_deref());
        if theirs != Some(build) {
            return Some(match theirs {
                Some(theirs) => label(&engine.version, Some(theirs)),
                None => format!("{} (another build)", engine.version),
            });
        }
    }
    replaced_executable(engine.pid)
        .then(|| format!("{} (an executable since replaced)", engine.version))
}

/// Linux: the engine runs a binary file that was replaced on disk (an upgrade in place).
fn replaced_executable(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        pid != std::process::id()
            && fs::read_link(format!("/proc/{pid}/exe"))
                .is_ok_and(|path| path.to_string_lossy().ends_with(" (deleted)"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        false
    }
}

/// Keep a matching engine; restart an idle mismatched one; refuse a busy mismatched one.
/// `Ok(None)` means it was stopped and the caller starts a fresh one.
fn reconcile(
    home: &std::path::Path,
    engine: Background,
    expected: Option<Expected<'_>>,
) -> Result<Option<Background>, Error> {
    let Some(old) = outdated(&engine, expected) else {
        return Ok(Some(engine));
    };
    let running = active_runs(&engine.client);
    if !running.is_empty() {
        return Err(Error::OutdatedBusy {
            version: old,
            running,
        });
    }
    stop_engine(home, &engine)?;
    set_notice(format!(
        "Restarted the background engine: it was jAIn {old}, now {}",
        expected.map_or_else(|| CLIENT_VERSION.to_owned(), |e| e.label())
    ));
    Ok(None)
}

/// Runs the engine is executing (running or pausing). Unknown is treated as none.
pub fn active_runs<T: crate::Transport>(client: &T) -> Vec<String> {
    use crate::transport::Request;
    let mut running = Vec::new();
    let mut after: Option<String> = None;
    for _ in 0..64 {
        let Ok(request) = Request::new(7, 0, &[]) else {
            break;
        };
        let request = request.query(&[("limit", "256")]);
        let request = match &after {
            Some(cursor) => request.query(&[("after", cursor)]),
            None => request,
        };
        let Ok(page) = client
            .execute(request)
            .and_then(|r| r.json::<jope_api::registry::RunPage>())
        else {
            break;
        };
        running.extend(
            page.runs
                .iter()
                .filter(|r| {
                    matches!(
                        r.state,
                        jope_api::RunState::Running | jope_api::RunState::Pausing
                    )
                })
                .map(|r| r.run_id.to_string()),
        );
        match page.next_after {
            Some(cursor) if after.as_ref() != Some(&cursor) => after = Some(cursor),
            _ => break,
        }
    }
    running
}

/// Ask the engine to stop and wait (at most ten seconds) for it to go.
fn stop_engine(home: &std::path::Path, engine: &Background) -> Result<(), Error> {
    let config = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(5)))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let token = record(home)?
        .filter(|r| r.pid == engine.pid)
        .map(|r| r.token)
        .ok_or(Error::InvalidMetadata)?;
    let request = ureq::http::Request::builder()
        .method("POST")
        .uri(format!("{}/__engine/stop", engine.url))
        .header("authorization", format!("Bearer {token}"))
        .body(Vec::new())
        .map_err(|_| Error::InvalidMetadata)?;
    let _ = agent.run(request);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let gone = record(home)
            .ok()
            .flatten()
            .is_none_or(|r| r.pid != engine.pid)
            || !alive(engine.pid);
        if gone {
            return Ok(());
        }
        thread::sleep(POLL);
    }
    Err(Error::Busy {
        pid: Some(engine.pid),
        command: format!(
            "jAIn engine {} did not stop; run `jain serve --stop --force`",
            engine.version
        ),
    })
}

mod local;
#[cfg(test)]
use local::detached;
use local::{alive, record};
use local::{
    attached, private_file, private_home, private_options, start_engine, store_owner, Reaper,
};

#[cfg(test)]
mod tests;
