//! Engine selection and bounded detached startup (spec §3.1).
//!
//! Callers supply the canonical `jope_paths::jopedime_home()` and their executable's
//! actual engine capability. The client's `engine` transport feature is a separate concern.
use crate::{http::Credentials, transport::Response, Http, Request};
use serde::Deserialize;
use std::{
    ffi::OsString,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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
pub enum Error {
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
impl Error {
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
impl fmt::Display for Error {
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
impl std::error::Error for Error {}
impl From<io::Error> for Error {
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

struct Reaper(Option<Child>);
impl Drop for Reaper {
    fn drop(&mut self) {
        if let Some(child) = self.0.take() {
            reap(child);
        }
    }
}

fn private_home(home: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(home)?;
    let metadata = fs::symlink_metadata(home)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "engine home must be a real private directory",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o022 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "engine home must not be writable by other accounts",
            ));
        }
    }
    Ok(())
}
fn private_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    options
}
fn private_file(file: &File) -> io::Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "daemon metadata must be a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "daemon metadata must have mode 0600",
            ));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "daemon metadata must not be a reparse point",
            ));
        }
    }
    Ok(())
}
fn record(home: &Path) -> Result<Option<Record>, Error> {
    let file = match private_options().open(home.join("daemon.json")) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    private_file(&file)?;
    if file.metadata()?.len() > MAX_METADATA {
        return Err(Error::InvalidMetadata);
    }
    let mut bytes = Vec::new();
    file.take(MAX_METADATA + 1).read_to_end(&mut bytes)?;
    if bytes.len() > MAX_METADATA as usize {
        return Err(Error::InvalidMetadata);
    }
    let Ok(record) = serde_json::from_slice::<Record>(&bytes) else {
        return Ok(None);
    };
    if !api_v1(&record.api_version) {
        return Err(Error::InvalidMetadata);
    }
    Ok(Some(record))
}
fn api_v1(version: &str) -> bool {
    version
        .strip_prefix("1.")
        .is_some_and(|minor| !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit()))
}
fn attached(home: &Path, deadline: Instant) -> Result<Option<Background>, Error> {
    let Some(record) = record(home)? else {
        return Ok(None);
    };
    if !alive(record.pid) {
        return Ok(None);
    }
    let credentials =
        Credentials::new(&record.url, &record.token).map_err(|_| Error::InvalidMetadata)?;
    // Never resolve DNS while applying the startup deadline or forward a local token elsewhere.
    let url = format!(
        "http://127.0.0.1:{}",
        credentials.authority.rsplit(':').next().unwrap()
    );
    let budget = deadline
        .saturating_duration_since(Instant::now())
        .min(Duration::from_millis(200));
    if budget.is_zero() {
        return Ok(None);
    }
    let wire = Credentials::new(&url, &record.token).map_err(|_| Error::InvalidMetadata)?;
    let config = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(budget))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let response = agent.run(
        wire.request(Request::new(1, 0, &[]).map_err(|_| Error::InvalidMetadata)?)
            .map_err(|_| Error::InvalidMetadata)?,
    );
    let Ok(response) = response else {
        return Ok(None);
    };
    let (parts, body) = response.into_parts();
    let info = Response::new(parts.status.as_u16(), parts.headers, body.into_reader())
        .json::<jope_api::SystemInfo>();
    let Ok(info) = info else { return Ok(None) };
    if Instant::now() >= deadline
        || !api_v1(&info.api_version)
        || self::record(home)?.as_ref() != Some(&record)
        || !alive(record.pid)
    {
        return Ok(None);
    }
    let client = Http::new(&url, &record.token).map_err(|_| Error::InvalidMetadata)?;
    Ok(Some(Background {
        client,
        url,
        pid: record.pid,
        version: record.version,
        api_version: record.api_version,
        started_at: record.started_at,
    }))
}
fn store_owner(home: &Path) -> io::Result<Option<(Option<u32>, String)>> {
    let mut file = match private_options().write(true).open(home.join("lock")) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    private_file(&file)?;
    match file.try_lock() {
        Ok(()) => Ok(None),
        Err(fs::TryLockError::WouldBlock) => {
            let mut text = String::new();
            // Windows may deny reading a range locked by another process.
            let _ = file.by_ref().take(4096).read_to_string(&mut text);
            let mut lines = text.lines();
            let pid = lines.next().and_then(|line| line.parse().ok());
            let command = lines.next().unwrap_or("unknown").to_owned();
            Ok(Some((pid, command)))
        }
        Err(fs::TryLockError::Error(e)) => Err(e),
    }
}
fn start_engine(options: &Options) -> io::Result<Child> {
    let mut command = Command::new(&options.executable);
    command
        .args(["serve", "--idle-exit", "30m"])
        .env("JOPEDIME_HOME", &options.home);
    detached(&mut command)?;
    command.spawn()
}
fn reap(mut child: Child) {
    let _ = thread::Builder::new()
        .name("jope-engine-reaper".into())
        .spawn(move || {
            let _ = child.wait();
        });
}
#[cfg(unix)]
#[allow(unsafe_code)]
fn detached(command: &mut Command) -> io::Result<()> {
    use std::os::unix::process::CommandExt;
    unsafe extern "C" {
        fn setsid() -> std::os::raw::c_int;
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: the post-fork hook only calls async-signal-safe setsid and reads errno.
    // It captures nothing, allocates nothing and takes no locks before exec.
    unsafe {
        command.pre_exec(|| {
            if setsid() == -1 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
    Ok(())
}
#[cfg(windows)]
fn detached(command: &mut Command) -> io::Result<()> {
    use std::os::windows::process::CommandExt;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(0x00000008 | 0x00000200);
    Ok(())
}
#[cfg(not(any(unix, windows)))]
fn detached(_: &mut Command) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "detached engine startup is unavailable",
    ))
}
#[cfg(unix)]
#[allow(unsafe_code)]
fn alive(pid: u32) -> bool {
    unsafe extern "C" {
        fn kill(pid: std::os::raw::c_int, signal: std::os::raw::c_int) -> std::os::raw::c_int;
    }
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal0 only queries an individual positive PID; it sends no signal.
    unsafe {
        kill(pid, 0) == 0 || io::Error::last_os_error().kind() == io::ErrorKind::PermissionDenied
    }
}
#[cfg(windows)]
#[allow(unsafe_code)]
fn alive(pid: u32) -> bool {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn GetExitCodeProcess(process: *mut c_void, code: *mut u32) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }
    if pid == 0 {
        return false;
    }
    // SAFETY: query-only access; a successful handle is closed exactly once.
    unsafe {
        let handle = OpenProcess(0x1000, 0, pid);
        if handle.is_null() {
            return io::Error::last_os_error().raw_os_error() == Some(5);
        }
        let mut code = 0;
        let ok = GetExitCodeProcess(handle, &mut code) != 0;
        CloseHandle(handle);
        ok && code == 259
    }
}
#[cfg(not(any(unix, windows)))]
fn alive(_: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        net::TcpListener,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const PROBE: &str = "discover::tests::native_client_probe";

    struct Home(PathBuf);
    impl Home {
        fn new() -> Self {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/client-discovery")
                .join(format!(
                    "{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
            private_home(&path).unwrap();
            Self(path)
        }
        fn options(&self) -> Options {
            Options {
                home: self.0.clone(),
                executable: self.0.join("missing-engine"),
                endpoint: None,
                mode: Mode::Interactive,
                engine_available: true,
            }
        }
    }
    impl Drop for Home {
        fn drop(&mut self) {
            // A remaining engine descriptor is retained for diagnosis, never removed underneath it.
            if !self.0.join("daemon.json").exists() {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
    }
    fn write_record(home: &Path, url: &str, pid: u32) {
        let bytes = serde_json::to_vec(&serde_json::json!({"url":url,"pid":pid,"token":TOKEN,
            "version":"test","api_version":"1.0","started_at":"test-epoch"}))
        .unwrap();
        let mut file = private_options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(home.join("daemon.json"))
            .unwrap();
        file.write_all(&bytes).unwrap();
    }
    fn select(options: &Options) -> Result<Engine, Error> {
        discover_with(options, None, Duration::from_millis(250), |_| {
            panic!("unexpected spawn")
        })
    }
    fn server(status: u16, delay: Duration) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut byte = [0];
            while !bytes.ends_with(b"\r\n\r\n") {
                assert_eq!(stream.read(&mut byte).unwrap(), 1);
                bytes.push(byte[0]);
            }
            let request = String::from_utf8(bytes).unwrap().to_ascii_lowercase();
            assert!(request.starts_with("get /v1/system http/1.1\r\n"));
            assert!(request.contains(&format!("authorization: bearer {TOKEN}\r\n")));
            thread::sleep(delay);
            let body = r#"{"version":"test","api_version":"1.0","health":"ok","local_paths":true,"capabilities":{"tasks":[],"languages":[],"devices":[],"learned_guidance":false},"warnings":[],"telemetry":[]}"#;
            let _ = write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nJope-Api-Version: 1.0\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        });
        (url, handle)
    }

    #[test]
    fn selection_precedence_and_batch_lite_contract() {
        let home = Home::new();
        let mut options = home.options();
        options.endpoint = Some("https://example.test:443".into());
        assert!(
            matches!(discover_with(&options, Some("bad".into()), STARTUP_TIMEOUT, |_| panic!()),
            Ok(Engine::Endpoint(url)) if url == "https://example.test:443")
        );
        options.endpoint = None;
        assert!(matches!(
            discover_with(
                &options,
                Some("http://localhost:9000".into()),
                STARTUP_TIMEOUT,
                |_| panic!()
            ),
            Ok(Engine::Endpoint(_))
        ));
        for url in [
            "",
            "file:///home",
            "http://a:0",
            "http://a/path",
            "http://token@a",
            "http://a?token=secret",
            "http://a#secret",
        ] {
            options.endpoint = Some(url.into());
            assert!(
                matches!(select(&options), Err(Error::InvalidEndpoint)),
                "{url}"
            );
        }
        options.endpoint = None;
        options.mode = Mode::Batch;
        assert!(matches!(select(&options), Ok(Engine::InProcess)));
        options.engine_available = false;
        let error = select(&options).unwrap_err();
        assert_eq!(error.to_string(), NO_ENGINE);
        assert_eq!(error.exit_code(), 9);
        assert!(!home.0.join(".discover.lock").exists());
    }

    #[test]
    fn authenticated_live_metadata_precedes_batch_and_lite() {
        let home = Home::new();
        let (url, handle) = server(200, Duration::ZERO);
        write_record(&home.0, &url, std::process::id());
        let mut options = home.options();
        options.engine_available = false;
        options.mode = Mode::Batch;
        let engine = select(&options).unwrap();
        assert!(
            matches!(&engine, Engine::Background(background) if background.pid == std::process::id())
        );
        assert!(!format!("{engine:?}").contains(TOKEN));
        handle.join().unwrap();
        fs::remove_file(home.0.join("daemon.json")).unwrap();
    }

    #[test]
    fn stale_malformed_and_refused_metadata_never_attach() {
        let home = Home::new();
        let mut options = home.options();
        options.engine_available = false;
        write_record(&home.0, "http://127.0.0.1:1", u32::MAX);
        assert!(matches!(select(&options), Err(Error::NoEngine)));
        fs::write(home.0.join("daemon.json"), b"{unfinished").unwrap();
        assert!(matches!(select(&options), Err(Error::NoEngine)));
        let (url, handle) = server(401, Duration::ZERO);
        write_record(&home.0, &url, std::process::id());
        assert!(matches!(select(&options), Err(Error::NoEngine)));
        handle.join().unwrap();
        fs::write(
            home.0.join("daemon.json"),
            vec![b' '; MAX_METADATA as usize + 1],
        )
        .unwrap();
        assert!(matches!(select(&options), Err(Error::InvalidMetadata)));
        fs::remove_file(home.0.join("daemon.json")).unwrap();
    }

    #[test]
    fn health_probe_obeys_deadline() {
        let home = Home::new();
        let (url, handle) = server(200, Duration::from_millis(400));
        write_record(&home.0, &url, std::process::id());
        let mut options = home.options();
        options.engine_available = false;
        let before = Instant::now();
        assert!(matches!(
            discover_with(&options, None, Duration::from_millis(80), |_| panic!()),
            Err(Error::NoEngine)
        ));
        assert!(before.elapsed() < Duration::from_millis(350));
        handle.join().unwrap();
        fs::remove_file(home.0.join("daemon.json")).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn public_metadata_and_symlinks_are_refused() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let home = Home::new();
        write_record(&home.0, "http://127.0.0.1:1", std::process::id());
        fs::set_permissions(
            home.0.join("daemon.json"),
            fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert!(matches!(select(&home.options()), Err(Error::Io(_))));
        fs::rename(home.0.join("daemon.json"), home.0.join("other")).unwrap();
        symlink(home.0.join("other"), home.0.join("daemon.json")).unwrap();
        assert!(matches!(select(&home.options()), Err(Error::Io(_))));
        fs::remove_file(home.0.join("daemon.json")).unwrap();
    }

    fn probe(home: &Home, engine: &Path, lite: bool) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                PROBE,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env_remove("JOPEDIME_ENDPOINT")
            .env("JOPE_CLT02_HOME", &home.0)
            .env("JOPE_CLT02_ENGINE", engine)
            .env("JOPE_CLT02_LITE", if lite { "1" } else { "0" })
            .env("TOKIO_WORKER_THREADS", "2")
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1");
        command
    }
    #[test]
    fn lite_process_exits_nine_with_documented_message() {
        let home = Home::new();
        let output = probe(&home, &home.options().executable, true)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(9));
        assert_eq!(String::from_utf8(output.stderr).unwrap().trim(), NO_ENGINE);
    }

    #[test]
    #[ignore = "subprocess fixture, invoked by supervised tests"]
    fn native_client_probe() {
        if let Ok(code) = std::env::var("JOPE_CLT02_EXIT") {
            std::process::exit(code.parse().unwrap());
        }
        let home =
            PathBuf::from(std::env::var_os("JOPE_CLT02_HOME").expect("supervised fixture home"));
        if let Some(gate) = std::env::var_os("JOPE_CLT02_GATE") {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !Path::new(&gate).exists() {
                assert!(Instant::now() < deadline, "parent never released clients");
                thread::sleep(POLL);
            }
        }
        let options = Options {
            home,
            executable: PathBuf::from(std::env::var_os("JOPE_CLT02_ENGINE").unwrap()),
            endpoint: None,
            mode: Mode::Interactive,
            engine_available: std::env::var("JOPE_CLT02_LITE").unwrap() != "1",
        };
        match discover(&options) {
            Ok(Engine::Background(background)) => println!(
                "DISCOVERY_RESULT {}",
                serde_json::json!({"pid":background.pid,"url":background.url})
            ),
            Ok(other) => panic!("unexpected selection: {other:?}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(error.exit_code().into());
            }
        }
    }

    #[test]
    fn losing_child_exit_is_not_readiness_and_retries_shared_metadata() {
        for exit in [0, 9] {
            let home = Home::new();
            let (url, handle) = server(200, Duration::ZERO);
            let directory = home.0.clone();
            let publisher = thread::spawn(move || {
                thread::sleep(Duration::from_millis(120));
                write_record(&directory, &url, std::process::id());
            });
            let result = discover_with(&home.options(), None, Duration::from_secs(2), |_| {
                // Both successful and unsuccessful child exits precede the winner's publication.
                let mut command = probe(&home, &home.options().executable, true);
                command.env("JOPE_CLT02_EXIT", exit.to_string());
                detached(&mut command)?;
                command.spawn()
            });
            assert!(matches!(result, Ok(Engine::Background(_))));
            publisher.join().unwrap();
            handle.join().unwrap();
            fs::remove_file(home.0.join("daemon.json")).unwrap();
        }
    }

    #[test]
    fn held_store_lock_reports_busy_without_modifying_it() {
        let home = Home::new();
        let mut file = private_options()
            .write(true)
            .create_new(true)
            .open(home.0.join("lock"))
            .unwrap();
        let payload = format!("{}\nbatch command\n", std::process::id());
        file.write_all(payload.as_bytes()).unwrap();
        file.lock().unwrap();
        let error = discover_with(&home.options(), None, Duration::from_millis(80), |_| {
            let mut command = probe(&home, &home.options().executable, true);
            command.env("JOPE_CLT02_EXIT", "9");
            detached(&mut command)?;
            command.spawn()
        })
        .unwrap_err();
        assert!(matches!(error, Error::Busy { .. }));
        assert_eq!(error.exit_code(), 9);
        drop(file);
        assert_eq!(fs::read_to_string(home.0.join("lock")).unwrap(), payload);
    }

    #[test]
    #[ignore = "requires JOPE_CLT02_ENGINE built from this source; native platform qualification"]
    fn ten_native_clients_share_one_detached_engine() {
        let executable = PathBuf::from(
            std::env::var_os("JOPE_CLT02_ENGINE").expect("real full engine binary required"),
        );
        assert!(executable.is_file());
        let home = Home::new();
        let gate = home.0.join("release-clients");
        let mut children: Vec<_> = (0..10)
            .map(|_| {
                probe(&home, &executable, false)
                    .env("JOPE_CLT02_GATE", &gate)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        fs::write(&gate, b"go").unwrap();
        let deadline = Instant::now() + Duration::from_secs(12);
        let mut results: Vec<serde_json::Value> = Vec::new();
        for child in &mut children {
            while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
                thread::sleep(POLL);
            }
        }
        let mut failures = Vec::new();
        for mut child in children {
            if child.try_wait().unwrap().is_none() {
                child.kill().unwrap();
            }
            let output = child.wait_with_output().unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            if !output.status.success() || !stdout.contains("1 passed; 0 failed") {
                failures.push(format!(
                    "status {} stdout {stdout} stderr {stderr}",
                    output.status
                ));
                continue;
            }
            let line = stdout
                .lines()
                .find_map(|line| line.split_once("DISCOVERY_RESULT ").map(|(_, json)| json));
            if let Some(result) = line.and_then(|line| serde_json::from_str(line).ok()) {
                results.push(result);
            } else {
                failures.push(format!("missing attachment inventory: {stdout}"));
            }
        }
        // Stop our fixture even when an assertion below will fail.
        let descriptor = record(&home.0).unwrap();
        let mut detached_session = true;
        if let Some(record) = &descriptor {
            assert!(
                alive(record.pid),
                "engine must outlive all client processes"
            );
            #[cfg(unix)]
            {
                detached_session = session(record.pid) == record.pid as i32;
            }
        }
        let stop = Command::new(&executable)
            .args(["serve", "--stop"])
            .env("JOPEDIME_HOME", &home.0)
            .env("TOKIO_WORKER_THREADS", "2")
            .output()
            .unwrap();
        let stop_deadline = Instant::now() + Duration::from_secs(5);
        while home.0.join("daemon.json").exists() && Instant::now() < stop_deadline {
            thread::sleep(POLL);
        }
        assert!(
            stop.status.success(),
            "fixture shutdown: {}",
            String::from_utf8_lossy(&stop.stderr)
        );
        assert!(
            !home.0.join("daemon.json").exists(),
            "fixture engine did not clean up"
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|result| result == &results[0]));
        assert_eq!(
            Some(results[0]["pid"].as_u64().unwrap()),
            descriptor.map(|r| r.pid as u64)
        );
        assert!(detached_session, "Unix engine must lead its own session");
        println!(
            "NATIVE_DISCOVERY {}",
            serde_json::json!({"platform":std::env::consts::OS,"clients":results,"detached_session":detached_session,"shutdown":true})
        );
    }
    #[cfg(unix)]
    #[allow(unsafe_code)]
    fn session(pid: u32) -> i32 {
        unsafe extern "C" {
            fn getsid(pid: i32) -> i32;
        }
        // SAFETY: read-only session query for our own fixture's positive PID.
        unsafe { getsid(pid as i32) }
    }
}
