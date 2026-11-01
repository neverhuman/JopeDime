//! Process ownership and lifecycle for the shared JopeDime engine.

use axum::{
    extract::{Extension, Request},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::post,
    Router,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};
use subtle::ConstantTimeEq;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::watch,
};

use crate::{
    config::Config,
    daemon_file::{self, DaemonFile, DaemonFileStatus},
    http::{
        auth::{BearerToken, LocalAuth},
        router,
    },
    store::Store,
};

const DEFAULT_IDLE_EXIT: Duration = Duration::from_secs(30 * 60);
const STOP_PATH: &str = "/__engine/stop";

/// Failure to own the engine or to start/stop its local listener.
#[derive(Debug)]
pub enum EngineError {
    Io(io::Error),
    Store(crate::store::StoreError),
    Busy { pid: Option<u32>, command: String },
    Daemon(String),
    InvalidIdleExit,
    Runtime(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "engine IO: {error}"),
            Self::Store(error) => write!(f, "engine store: {error}"),
            Self::Busy { pid, command } => write!(
                f,
                "ENGINE_BUSY: pid={} command={}",
                pid.map_or_else(|| "unknown".to_owned(), |pid| pid.to_string()),
                if command.is_empty() {
                    "unknown"
                } else {
                    command
                }
            ),
            Self::Daemon(message) => write!(f, "engine daemon: {message}"),
            Self::InvalidIdleExit => {
                f.write_str("idle exit must be a positive duration (for example 30m or 2s)")
            }
            Self::Runtime(message) => write!(f, "engine runtime: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<io::Error> for EngineError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::store::StoreError> for EngineError {
    fn from(error: crate::store::StoreError) -> Self {
        Self::Store(error)
    }
}

/// A live process's OS advisory lock. Dropping the file releases ownership.
pub struct EngineLock {
    file: File,
}

impl std::fmt::Debug for EngineLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineLock").finish_non_exhaustive()
    }
}

impl EngineLock {
    pub fn acquire(home: impl AsRef<Path>) -> Result<Self, EngineError> {
        let home = home.as_ref();
        if !home.exists() {
            fs::create_dir_all(home)?;
        }
        if !fs::symlink_metadata(home)?.file_type().is_dir() {
            return Err(EngineError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "engine home is not a real directory",
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(home, fs::Permissions::from_mode(0o700))?;
        }
        let path = home.join("lock");
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_file() {
                return Err(EngineError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "engine lock is not a regular file",
                )));
            }
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(&path)?;
        if let Err(error) = file.try_lock() {
            let mut holder = read_holder(&path);
            for _ in 0..20 {
                if holder.0.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
                holder = read_holder(&path);
            }
            return match error {
                std::fs::TryLockError::WouldBlock => Err(EngineError::Busy {
                    pid: holder.0,
                    command: holder.1,
                }),
                std::fs::TryLockError::Error(error) => Err(EngineError::Io(error)),
            };
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        writeln!(file, "{}", std::process::id())?;
        writeln!(file, "{}", process_command())?;
        file.sync_all()?;
        Ok(Self { file })
    }

    pub fn is_held(&self) -> bool {
        self.file.metadata().is_ok()
    }
}

fn process_command() -> String {
    let command = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    command
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(1024)
        .collect()
}

fn read_holder(path: &Path) -> (Option<u32>, String) {
    let mut text = String::new();
    if File::open(path)
        .and_then(|file| file.take(4096).read_to_string(&mut text))
        .is_err()
    {
        return (None, String::new());
    }
    let mut lines = text.lines();
    (
        lines.next().and_then(|line| line.parse().ok()),
        lines.next().unwrap_or_default().to_owned(),
    )
}

/// The same home lock and recovery path used by background engines, without daemon publication.
pub struct InProcessEngine {
    // Fields drop in declaration order: close redb before releasing home ownership.
    store: Arc<Store>,
    home: PathBuf,
    executable: PathBuf,
    runs: Mutex<Option<Arc<crate::http::runs::RunService>>>,
    activity: Arc<EngineActivity>,
    _lock: Arc<EngineLock>,
}

/// Read-only request state lent from the one live engine/store instance.
#[derive(Clone)]
pub(crate) struct DatasetState {
    pub store: Arc<Store>,
    pub config: Config,
    pub home: PathBuf,
}

impl InProcessEngine {
    pub fn open_default() -> Result<Self, EngineError> {
        Self::open(jope_paths::jopedime_home())
    }
    pub fn open(home: impl AsRef<Path>) -> Result<Self, EngineError> {
        Self::open_with_worker(home, std::env::current_exe()?)
    }

    /// Trusted embeddings supply their authenticated appliance executable, which
    /// implements the existing worker protocol. This is not a remote API input.
    pub fn open_with_worker(
        home: impl AsRef<Path>,
        executable: impl AsRef<Path>,
    ) -> Result<Self, EngineError> {
        #[cfg(unix)]
        let ranker = crate::proposal::from_self().map_err(|error| {
            EngineError::Runtime(format!("embedded proposal integrity: {error:#}"))
        })?;
        #[cfg(not(unix))]
        let ranker: Option<crate::run::learned::AdmittedRanker> = None;
        let home = home.as_ref();
        let lock = Arc::new(EngineLock::acquire(home)?);
        let store = Arc::new(Store::open(home)?);
        Ok(Self {
            store,
            home: home.canonicalize()?,
            executable: executable.as_ref().to_path_buf(),
            runs: Mutex::new(None),
            activity: Arc::new(EngineActivity {
                ranker: ranker.map(Arc::new),
                ..EngineActivity::default()
            }),
            _lock: lock,
        })
    }

    /// Explicit trusted-host admission for qualification of the signed frozen
    /// candidate. Ordinary engines and Auto guidance retain their quality gates.
    pub fn open_with_ranker(
        home: impl AsRef<Path>,
        executable: impl AsRef<Path>,
        ranker: crate::run::learned::AdmittedRanker,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::open_with_worker(home, executable)?;
        engine.activity = Arc::new(EngineActivity {
            ranker: Some(Arc::new(ranker)),
            ..EngineActivity::default()
        });
        Ok(engine)
    }

    /// The same store-bound router serves embedded clients and loopback HTTP.
    pub fn router(&self, auth: LocalAuth) -> Result<Router, EngineError> {
        self.router_with_activity(auth, Arc::clone(&self.activity))
    }

    fn router_with_activity(
        &self,
        auth: LocalAuth,
        activity: Arc<EngineActivity>,
    ) -> Result<Router, EngineError> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| EngineError::Runtime("run service lock poisoned".into()))?;
        if runs.is_none() {
            let environment = std::env::vars().collect();
            let config = Config::load(
                &self.home.join("config.toml"),
                &Default::default(),
                &environment,
            )
            .map_err(|e| EngineError::Runtime(e.to_string()))?;
            *runs = Some(Arc::new(
                crate::http::runs::RunService::new(
                    Arc::clone(&self.store),
                    config,
                    self.home.clone(),
                    self.executable.clone(),
                    activity,
                    Arc::clone(&self._lock),
                )
                .map_err(|e| EngineError::Runtime(e.to_string()))?,
            ));
        }
        let service = Arc::clone(runs.as_ref().expect("run service initialized"));
        Ok(router::router_with_state(
            auth,
            DatasetState {
                store: self.store_handle(),
                config: service.config().clone(),
                home: self.home.clone(),
            },
            Some(service),
        ))
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub(crate) fn store_handle(&self) -> Arc<Store> {
        Arc::clone(&self.store)
    }
}

/// Tracks run lifetimes and accepted TCP clients for idle shutdown.
pub struct EngineActivity {
    counts: Mutex<ActivityCounts>,
    ranker: Option<Arc<crate::run::learned::AdmittedRanker>>,
}

struct ActivityCounts {
    active_runs: usize,
    connected_clients: usize,
    last_activity: tokio::time::Instant,
}

impl Default for EngineActivity {
    fn default() -> Self {
        Self {
            ranker: None,
            counts: Mutex::new(ActivityCounts {
                active_runs: 0,
                connected_clients: 0,
                last_activity: tokio::time::Instant::now(),
            }),
        }
    }
}

impl EngineActivity {
    pub(crate) fn admitted_ranker(&self) -> Option<&crate::run::learned::AdmittedRanker> {
        self.ranker.as_deref()
    }
    pub(crate) fn active_runs(&self) -> usize {
        self.counts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .active_runs
    }
    pub fn begin_run(self: &Arc<Self>) -> ActivityGuard {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.active_runs += 1;
        counts.last_activity = tokio::time::Instant::now();
        drop(counts);
        ActivityGuard {
            activity: Arc::clone(self),
            kind: ActivityKind::Run,
        }
    }

    fn connected_client(self: &Arc<Self>) -> ActivityGuard {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.connected_clients += 1;
        counts.last_activity = tokio::time::Instant::now();
        drop(counts);
        ActivityGuard {
            activity: Arc::clone(self),
            kind: ActivityKind::Client,
        }
    }

    fn idle_for(&self, duration: Duration) -> bool {
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.active_runs == 0
            && counts.connected_clients == 0
            && counts.last_activity.elapsed() >= duration
    }
}

enum ActivityKind {
    Run,
    Client,
}

/// Decrements the associated activity count when the run or client ends.
pub struct ActivityGuard {
    activity: Arc<EngineActivity>,
    kind: ActivityKind,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let mut counts = self
            .activity
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match self.kind {
            ActivityKind::Run => counts.active_runs = counts.active_runs.saturating_sub(1),
            ActivityKind::Client => {
                counts.connected_clients = counts.connected_clients.saturating_sub(1)
            }
        }
        counts.last_activity = tokio::time::Instant::now();
    }
}

struct TrackedListener {
    listener: TcpListener,
    activity: Arc<EngineActivity>,
}

struct TrackedStream {
    stream: TcpStream,
    _activity: ActivityGuard,
}

impl AsyncRead for TrackedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

impl axum::serve::Listener for TrackedListener {
    type Io = TrackedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.listener.accept().await {
                Ok((stream, address)) => {
                    return (
                        TrackedStream {
                            stream,
                            _activity: self.activity.connected_client(),
                        },
                        address,
                    );
                }
                Err(error) => {
                    tokio::time::sleep(if error.kind() == io::ErrorKind::Interrupted {
                        Duration::ZERO
                    } else {
                        Duration::from_millis(100)
                    })
                    .await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

struct StopState {
    token: String,
    port: u16,
    shutdown: watch::Sender<bool>,
}

async fn request_stop(
    Extension(state): Extension<Arc<StopState>>,
    request: Request,
) -> impl IntoResponse {
    let expected_host = format!("127.0.0.1:{}", state.port);
    let headers = request.headers();
    let host_ok = single_header(headers, header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value == expected_host
                || value.eq_ignore_ascii_case(&format!("localhost:{}", state.port))
        });
    let uri_authority_ok = request.uri().authority().is_none_or(|authority| {
        authority.as_str() == expected_host
            || authority
                .as_str()
                .eq_ignore_ascii_case(&format!("localhost:{}", state.port))
    });
    let auth_ok = single_header(headers, header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .is_some_and(|(scheme, value)| {
            scheme.eq_ignore_ascii_case("Bearer")
                && bool::from(value.as_bytes().ct_eq(state.token.as_bytes()))
        });
    if headers.contains_key(header::ORIGIN) || !host_ok || !uri_authority_ok || !auth_ok {
        return StatusCode::UNAUTHORIZED;
    }
    let _ = state.shutdown.send(true);
    StatusCode::NO_CONTENT
}

fn single_header(headers: &HeaderMap, name: HeaderName) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

/// Start the shared engine in this process. Store recovery occurs after lock acquisition.
pub fn serve(idle_exit: Option<Duration>) -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    let engine = InProcessEngine::open(&home)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| EngineError::Runtime(error.to_string()))?;
    runtime.block_on(serve_locked(
        home,
        engine,
        idle_exit.unwrap_or(DEFAULT_IDLE_EXIT),
    ))
}

async fn serve_locked(
    home: PathBuf,
    engine: InProcessEngine,
    idle_exit: Duration,
) -> Result<(), EngineError> {
    let activity = Arc::clone(&engine.activity);
    serve_with_activity(home, engine, idle_exit, activity).await
}

async fn serve_with_activity(
    home: PathBuf,
    engine: InProcessEngine,
    idle_exit: Duration,
    activity: Arc<EngineActivity>,
) -> Result<(), EngineError> {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    let address = listener.local_addr()?;
    // The socket is bound before the descriptor is atomically published.
    let record = daemon_file::write_daemon_file(
        &home,
        format!("http://127.0.0.1:{}", address.port()),
        env!("CARGO_PKG_VERSION"),
        "1.0",
        chrono::Utc::now().to_rfc3339(),
    )?;
    let bearer = BearerToken::from_hex(&record.token)
        .map_err(|error| EngineError::Daemon(error.to_string()))?;
    let auth = LocalAuth::new(bearer, address.port())
        .map_err(|error| EngineError::Daemon(error.to_string()))?;
    let (shutdown, mut shutdown_rx) = watch::channel(false);
    let state = Arc::new(StopState {
        token: record.token.clone(),
        port: address.port(),
        shutdown,
    });
    let app: Router = engine
        .router_with_activity(auth, Arc::clone(&activity))?
        .route(STOP_PATH, post(request_stop))
        .layer(Extension(state))
        .layer(Extension(Arc::clone(&activity)));
    let tracked = TrackedListener {
        listener,
        activity: Arc::clone(&activity),
    };
    let server = axum::serve(tracked, app).with_graceful_shutdown(async move {
        let idle = tokio::time::interval(Duration::from_millis(100));
        tokio::pin!(idle);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() { break; }
                }
                _ = idle.tick() => {
                    if activity.idle_for(idle_exit) {
                        break;
                    }
                }
            }
        }
    });
    let result = server
        .await
        .map_err(|error| EngineError::Runtime(error.to_string()));
    let cleanup = daemon_file::remove_daemon_file(&home).map_err(EngineError::Io);
    cleanup?;
    result
}

/// Read daemon status or send an authenticated stop request. Never returns the bearer token.
pub fn run_cli(status: bool, stop: bool, idle_exit: Option<&str>) -> Result<(), EngineError> {
    if status {
        return show_status();
    }
    if stop {
        return stop_daemon();
    }
    let idle = idle_exit
        .map(parse_duration)
        .transpose()?
        .unwrap_or(DEFAULT_IDLE_EXIT);
    serve(Some(idle))
}

fn show_status() -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    match daemon_file::read_daemon_file(&home) {
        Ok(DaemonFileStatus::Running(record)) | Ok(DaemonFileStatus::Unknown(record)) => {
            let reachable = request_daemon(&record, "GET", "/v1/system").is_ok();
            println!(
                "{} (pid {})",
                if reachable { "running" } else { "unreachable" },
                record.pid
            );
            Ok(())
        }
        Ok(DaemonFileStatus::Stale(record)) => {
            println!("stopped (stale daemon record for pid {})", record.pid);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("stopped");
            Ok(())
        }
        Err(error) => Err(EngineError::Io(error)),
    }
}

fn stop_daemon() -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    let record = match daemon_file::read_daemon_file(&home) {
        Ok(DaemonFileStatus::Running(record) | DaemonFileStatus::Unknown(record)) => record,
        Ok(DaemonFileStatus::Stale(_)) => {
            return Err(EngineError::Daemon("engine is not running".into()));
        }
        Err(error) => return Err(EngineError::Io(error)),
    };
    request_daemon(&record, "POST", STOP_PATH)?;
    println!("stop requested for engine pid {}", record.pid);
    Ok(())
}

fn request_daemon(record: &DaemonFile, method: &str, path: &str) -> Result<(), EngineError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| EngineError::Runtime(error.to_string()))?;
    runtime.block_on(request_daemon_async(record, method, path))
}

async fn request_daemon_async(
    record: &DaemonFile,
    method: &str,
    path: &str,
) -> Result<(), EngineError> {
    let authority = record
        .url
        .strip_prefix("http://")
        .ok_or_else(|| EngineError::Daemon("daemon URL is not loopback HTTP".into()))?;
    if !authority.starts_with("127.0.0.1:") {
        return Err(EngineError::Daemon("daemon URL is not loopback".into()));
    }
    let port = authority
        .rsplit(':')
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| EngineError::Daemon("invalid daemon port".into()))?;
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).await?;
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
            record.token
        );
        stream.write_all(request.as_bytes()).await?;
        let mut response = Vec::new();
        stream.take(4096).read_to_end(&mut response).await?;
        let status = std::str::from_utf8(&response)
            .ok()
            .and_then(|text| text.lines().next())
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok());
        if status.is_some_and(|code| (200..300).contains(&code)) {
            Ok(())
        } else {
            Err(EngineError::Daemon(format!(
                "request failed (HTTP {})",
                status.map_or("unknown".into(), |code| code.to_string())
            )))
        }
    })
    .await
    .map_err(|_| EngineError::Daemon("request timed out".into()))?
}

fn parse_duration(value: &str) -> Result<Duration, EngineError> {
    let split = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    let number = number
        .parse::<u64>()
        .map_err(|_| EngineError::InvalidIdleExit)?;
    let multiplier = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => return Err(EngineError::InvalidIdleExit),
    };
    let seconds = number
        .checked_mul(multiplier)
        .filter(|seconds| *seconds > 0)
        .ok_or(EngineError::InvalidIdleExit)?;
    Ok(Duration::from_secs(seconds))
}

/// Utility for verifying that a run remains active until its owner drops the guard.
#[cfg(test)]
fn _active_run(activity: &Arc<EngineActivity>) -> ActivityGuard {
    activity.begin_run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn only_one_process_can_hold_the_home_lock_and_busy_reports_its_owner() {
        let home = tempfile::tempdir().unwrap();
        let owner = EngineLock::acquire(home.path()).unwrap();
        let busy = EngineLock::acquire(home.path()).unwrap_err();
        assert!(
            matches!(busy, EngineError::Busy { pid: Some(pid), .. } if pid == std::process::id())
        );
        assert!(owner.is_held());
    }

    #[cfg(unix)]
    #[test]
    fn lock_refuses_a_symlink_instead_of_following_it() {
        use std::os::unix::fs::symlink;

        let home = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let target = other.path().join("target");
        fs::write(&target, "untouched").unwrap();
        symlink(&target, home.path().join("lock")).unwrap();
        assert!(EngineLock::acquire(home.path()).is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "untouched");
    }

    #[test]
    fn in_process_engine_takes_the_same_lock_and_opens_recovered_store_without_daemon_file() {
        let home = tempfile::tempdir().unwrap();
        let engine = InProcessEngine::open(home.path()).unwrap();
        assert!(engine._lock.is_held());
        assert!(!home.path().join("daemon.json").exists());
        assert!(EngineLock::acquire(home.path()).is_err());
    }

    #[test]
    fn active_runs_and_open_tcp_clients_prevent_idle_exit() {
        let activity = Arc::new(EngineActivity::default());
        assert!(!activity.idle_for(Duration::from_secs(1)));
        let run = _active_run(&activity);
        assert!(!activity.idle_for(Duration::from_secs(1)));
        drop(run);
        assert!(!activity.idle_for(Duration::from_secs(1)));
        let client = activity.connected_client();
        assert!(!activity.idle_for(Duration::ZERO));
        drop(client);
        assert!(!activity.idle_for(Duration::from_secs(1)));
    }

    #[test]
    fn idle_exit_duration_parser_is_bounded_and_requires_units() {
        assert_eq!(parse_duration("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_duration("30m").unwrap(), Duration::from_secs(1800));
        assert!(parse_duration("0s").is_err());
        assert!(parse_duration("2").is_err());
        assert!(parse_duration("-1m").is_err());
    }

    #[tokio::test]
    async fn bound_engine_publishes_then_authenticated_stop_shuts_it_down() {
        let home = tempfile::tempdir().unwrap();
        let engine = InProcessEngine::open(home.path()).unwrap();
        let server_home = home.path().to_path_buf();
        let server = tokio::spawn(serve_locked(
            server_home.clone(),
            engine,
            Duration::from_secs(60),
        ));
        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(DaemonFileStatus::Running(record)) =
                    daemon_file::read_daemon_file(&server_home)
                {
                    if request_daemon_async(&record, "GET", "/v1/system")
                        .await
                        .is_ok()
                    {
                        break record;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        request_daemon_async(&record, "POST", STOP_PATH)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(!server_home.join("daemon.json").exists());
    }

    #[tokio::test]
    async fn unauthorized_stop_cannot_signal_shutdown() {
        let home = tempfile::tempdir().unwrap();
        let engine = InProcessEngine::open(home.path()).unwrap();
        let server_home = home.path().to_path_buf();
        let server = tokio::spawn(serve_locked(
            server_home.clone(),
            engine,
            Duration::from_secs(60),
        ));
        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(DaemonFileStatus::Running(record)) =
                    daemon_file::read_daemon_file(&server_home)
                {
                    break record;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let authority = record.url.strip_prefix("http://").unwrap();
        let port = authority
            .rsplit(':')
            .next()
            .unwrap()
            .parse::<u16>()
            .unwrap();
        let mut stream = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
            .await
            .unwrap();
        stream.write_all(format!("POST {STOP_PATH} HTTP/1.1\r\nHost: {authority}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n", "00".repeat(32)).as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 401"));
        assert!(!server.is_finished());
        let record = daemon_file::read_daemon_file(&server_home).unwrap();
        let DaemonFileStatus::Running(record) = record else {
            panic!("daemon stopped unexpectedly")
        };
        request_daemon_async(&record, "POST", STOP_PATH)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn two_second_idle_exit_waits_for_active_runs_and_connected_clients() {
        let home = tempfile::tempdir().unwrap();
        let engine = InProcessEngine::open(home.path()).unwrap();
        let activity = Arc::new(EngineActivity::default());
        let server_home = home.path().to_path_buf();
        let server_activity = Arc::clone(&activity);
        let server = tokio::spawn(serve_with_activity(
            server_home.clone(),
            engine,
            Duration::from_secs(2),
            server_activity,
        ));
        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(DaemonFileStatus::Running(record)) =
                    daemon_file::read_daemon_file(&server_home)
                {
                    break record;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();

        let active_run = activity.begin_run();
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert!(!server.is_finished(), "active run must suppress idle exit");
        drop(active_run);

        let authority = record.url.strip_prefix("http://").unwrap();
        let port = authority
            .rsplit(':')
            .next()
            .unwrap()
            .parse::<u16>()
            .unwrap();
        let client = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            !server.is_finished(),
            "connected client must suppress idle exit"
        );
        drop(client);

        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(!server_home.join("daemon.json").exists());
    }

    #[test]
    fn simultaneous_lock_attempts_have_one_winner() {
        let home = tempfile::tempdir().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let path = home.path().to_path_buf();
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                EngineLock::acquire(path)
            }));
        }
        barrier.wait();
        let results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(EngineError::Busy { .. })))
                .count(),
            1
        );
    }

    #[test]
    fn in_process_handoff_retains_ownership_through_store_close() {
        let home = tempfile::tempdir().unwrap();
        drop(InProcessEngine::open(home.path()).unwrap());
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let home = home.path().to_path_buf();
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                let mut starts = 0;
                let mut unexpected = Vec::new();
                while std::time::Instant::now() < deadline {
                    match InProcessEngine::open(&home) {
                        Ok(engine) => {
                            starts += 1;
                            drop(engine);
                        }
                        Err(EngineError::Busy { .. }) => {}
                        Err(error) => unexpected.push(error.to_string()),
                    }
                }
                (starts, unexpected)
            }));
        }
        barrier.wait();
        let mut starts = 0;
        let mut unexpected = Vec::new();
        for thread in threads {
            let (count, errors) = thread.join().unwrap();
            starts += count;
            unexpected.extend(errors);
        }
        assert!(starts > 0, "the probe must open an engine");
        assert!(
            unexpected.is_empty(),
            "advisory ownership ended before Store closed: {unexpected:?}"
        );
    }
}
