//! Read and write the owner-scoped `daemon.json` record.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const DAEMON_FILE_NAME: &str = "daemon.json";
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The fields shared by a local engine and its clients through `daemon.json`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DaemonFile {
    pub url: String,
    pub token: String,
    pub pid: u32,
    pub version: String,
    pub api_version: String,
    pub started_at: String,
}

/// The record plus what this process can establish about its owner PID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DaemonFileStatus {
    Running(DaemonFile),
    Stale(DaemonFile),
    Unknown(DaemonFile),
}

/// Create a fresh record for this process, generate its 256-bit bearer token,
/// and atomically write it under the resolved JopeDime home directory.
///
/// `home` should be the directory returned by `jope_paths::jopedime_home()`.
pub fn write_daemon_file(
    home: &Path,
    url: impl Into<String>,
    version: impl Into<String>,
    api_version: impl Into<String>,
    started_at: impl Into<String>,
) -> io::Result<DaemonFile> {
    let mut random = [0_u8; 32];
    fill_random(&mut random)?;
    let record = DaemonFile {
        url: url.into(),
        token: encode_hex(&random),
        pid: std::process::id(),
        version: version.into(),
        api_version: api_version.into(),
        started_at: started_at.into(),
    };
    write_record(home, &record)?;
    Ok(record)
}

/// Read `daemon.json` and classify the recorded PID where the platform allows it.
pub fn read_daemon_file(home: &Path) -> io::Result<DaemonFileStatus> {
    let bytes = fs::read(daemon_file_path(home))?;
    let record: DaemonFile = serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(match process_liveness(record.pid) {
        ProcessLiveness::Running => DaemonFileStatus::Running(record),
        ProcessLiveness::Stale => DaemonFileStatus::Stale(record),
        ProcessLiveness::Unknown => DaemonFileStatus::Unknown(record),
    })
}

/// Remove `daemon.json` during clean engine shutdown. Missing files are harmless.
pub fn remove_daemon_file(home: &Path) -> io::Result<()> {
    match fs::remove_file(daemon_file_path(home)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn daemon_file_path(home: &Path) -> PathBuf {
    home.join(DAEMON_FILE_NAME)
}

fn write_record(home: &Path, record: &DaemonFile) -> io::Result<()> {
    fs::create_dir_all(home)?;
    let destination = daemon_file_path(home);
    let mut last_collision = None;

    for _ in 0..32 {
        let temporary = temporary_path(home);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };

        let result = write_json(&mut file, record).and_then(|()| file.sync_all());
        drop(file);
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }

        if let Err(error) = fs::rename(&temporary, &destination) {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        return Ok(());
    }

    Err(last_collision.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a temporary daemon.json file",
        )
    }))
}

fn write_json(writer: &mut impl Write, record: &DaemonFile) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, record).map_err(|error| {
        let kind = error.io_error_kind().unwrap_or(io::ErrorKind::InvalidData);
        io::Error::new(kind, error)
    })?;
    writer.write_all(b"\n")
}

fn temporary_path(home: &Path) -> PathBuf {
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    home.join(format!(
        ".{DAEMON_FILE_NAME}.{}.{}.{}.tmp",
        std::process::id(),
        time,
        sequence
    ))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(unix)]
fn fill_random(bytes: &mut [u8]) -> io::Result<()> {
    File::open("/dev/urandom")?.read_exact(bytes)
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn fill_random(bytes: &mut [u8]) -> io::Result<()> {
    use std::ffi::c_void;

    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(algorithm: *mut c_void, buffer: *mut u8, length: u32, flags: u32)
            -> i32;
    }

    const USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;
    let length = u32::try_from(bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "random buffer too large"))?;
    // SAFETY: BCryptGenRandom writes at most `length` bytes to the mutable slice supplied here.
    let status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            bytes.as_mut_ptr(),
            length,
            USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if status >= 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status))
    }
}

#[cfg(not(any(unix, windows)))]
fn fill_random(_bytes: &mut [u8]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "secure random source is unavailable on this platform",
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessLiveness {
    Running,
    Stale,
    Unknown,
}

#[cfg(unix)]
fn process_liveness(pid: u32) -> ProcessLiveness {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    if pid == 0 || pid > i32::MAX as u32 {
        return ProcessLiveness::Stale;
    }
    match kill(Pid::from_raw(pid as i32), None) {
        Ok(()) | Err(Errno::EPERM) => ProcessLiveness::Running,
        Err(Errno::ESRCH) => ProcessLiveness::Stale,
        Err(_) => ProcessLiveness::Unknown,
    }
}

#[cfg(not(unix))]
fn process_liveness(_pid: u32) -> ProcessLiveness {
    ProcessLiveness::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(home: &Path) -> DaemonFile {
        write_daemon_file(
            home,
            "http://127.0.0.1:43123",
            "0.4.1",
            "1.0",
            "2026-10-30T10:00:00Z",
        )
        .expect("write daemon file")
    }

    #[test]
    fn daemon_file_round_trips_and_records_a_256_bit_hex_token() {
        let home = tempfile::tempdir().expect("temporary home");
        let written = sample(home.path());

        assert_eq!(written.token.len(), 64);
        assert!(written
            .token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(
            read_daemon_file(home.path()).expect("read daemon file"),
            DaemonFileStatus::Running(written)
        );
    }

    #[cfg(unix)]
    #[test]
    fn daemon_file_permissions_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let home = tempfile::tempdir().expect("temporary home");
        sample(home.path());
        let mode = fs::metadata(daemon_file_path(home.path()))
            .expect("daemon metadata")
            .permissions()
            .mode()
            & 0o777;

        assert_eq!(mode, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn daemon_file_with_dead_pid_is_reported_stale() {
        let home = tempfile::tempdir().expect("temporary home");
        let record = DaemonFile {
            url: "http://127.0.0.1:43123".to_owned(),
            token: "00".repeat(32),
            pid: u32::MAX,
            version: "0.4.1".to_owned(),
            api_version: "1.0".to_owned(),
            started_at: "2026-10-30T10:00:00Z".to_owned(),
        };
        write_record(home.path(), &record).expect("write stale daemon file");

        assert_eq!(
            read_daemon_file(home.path()).expect("read stale daemon file"),
            DaemonFileStatus::Stale(record)
        );
    }

    #[test]
    fn clean_shutdown_removes_the_daemon_file_idempotently() {
        let home = tempfile::tempdir().expect("temporary home");
        sample(home.path());

        remove_daemon_file(home.path()).expect("remove daemon file");
        remove_daemon_file(home.path()).expect("missing file is harmless");
        assert!(!daemon_file_path(home.path()).exists());
    }

    #[test]
    fn a_second_start_atomically_replaces_the_previous_record() {
        let home = tempfile::tempdir().expect("temporary home");
        let first = sample(home.path());
        let second = write_daemon_file(
            home.path(),
            "http://127.0.0.1:43124",
            "0.4.1",
            "1.0",
            "2026-10-30T10:01:00Z",
        )
        .expect("replace daemon file");

        assert_ne!(first.token, second.token);
        assert_eq!(
            read_daemon_file(home.path()).expect("read replacement"),
            DaemonFileStatus::Running(second)
        );
    }

    #[cfg(not(unix))]
    #[test]
    fn non_unix_pid_liveness_is_unknown() {
        let home = tempfile::tempdir().expect("temporary home");
        let record = sample(home.path());

        assert_eq!(
            read_daemon_file(home.path()).expect("read daemon file"),
            DaemonFileStatus::Unknown(record)
        );
    }
}
