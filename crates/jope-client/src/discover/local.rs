use super::*;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
};

pub(super) struct Reaper(pub(super) Option<Child>);
impl Drop for Reaper {
    fn drop(&mut self) {
        if let Some(child) = self.0.take() {
            reap(child);
        }
    }
}

pub(super) fn private_home(home: &Path) -> io::Result<()> {
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
pub(super) fn private_options() -> OpenOptions {
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
pub(super) fn private_file(file: &File) -> io::Result<()> {
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
pub(super) fn record(home: &Path) -> Result<Option<Record>, Error> {
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
pub(super) fn attached(home: &Path, deadline: Instant) -> Result<Option<Background>, Error> {
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
pub(super) fn store_owner(home: &Path) -> io::Result<Option<(Option<u32>, String)>> {
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
pub(super) fn start_engine(options: &Options) -> io::Result<Child> {
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
pub(super) fn detached(command: &mut Command) -> io::Result<()> {
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
pub(super) fn detached(command: &mut Command) -> io::Result<()> {
    use std::os::windows::process::CommandExt;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(0x00000008 | 0x00000200);
    Ok(())
}
#[cfg(not(any(unix, windows)))]
pub(super) fn detached(_: &mut Command) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "detached engine startup is unavailable",
    ))
}
#[cfg(unix)]
#[allow(unsafe_code)]
pub(super) fn alive(pid: u32) -> bool {
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
pub(super) fn alive(pid: u32) -> bool {
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
pub(super) fn alive(_: u32) -> bool {
    false
}
