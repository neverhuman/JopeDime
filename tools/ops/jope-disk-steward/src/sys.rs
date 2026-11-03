//! Low-level POSIX/Linux system calls, flock guards, statvfs, and atomic file I/O.

use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const LOCK_EX: i32 = 2;
pub const LOCK_NB: i32 = 4;
pub const LOCK_UN: i32 = 8;

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_CREAT: i32 = 0o100;
pub const O_EXCL: i32 = 0o200;
pub const O_NOFOLLOW: i32 = 0o400000;
pub const O_CLOEXEC: i32 = 0o2000000;
pub const O_NONBLOCK: i32 = 0o4000;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Statvfs {
    pub f_bsize: u64,
    pub f_frsize: u64,
    pub f_blocks: u64,
    pub f_bfree: u64,
    pub f_bavail: u64,
    pub f_files: u64,
    pub f_ffree: u64,
    pub f_favail: u64,
    pub f_fsid: u64,
    pub f_flag: u64,
    pub f_namemax: u64,
    pub __f_spare: [i32; 6],
}

extern "C" {
    pub fn flock(fd: i32, operation: i32) -> i32;
    pub fn statvfs(path: *const i8, buf: *mut Statvfs) -> i32;
    pub fn getuid() -> u32;
    pub fn getgid() -> u32;
    pub fn getpid() -> i32;
    pub fn open(path: *const i8, oflag: i32, mode: u32) -> i32;
    pub fn close(fd: i32) -> i32;
    pub fn fsync(fd: i32) -> i32;
    pub fn ftruncate(fd: i32, length: i64) -> i32;
    pub fn posix_fallocate(fd: i32, offset: i64, len: i64) -> i32;
    pub fn inotify_init1(flags: i32) -> i32;
    pub fn inotify_add_watch(fd: i32, pathname: *const i8, mask: u32) -> i32;
}

pub fn current_uid() -> u32 {
    unsafe { getuid() }
}

pub fn current_gid() -> u32 {
    unsafe { getgid() }
}

pub fn current_pid() -> i32 {
    unsafe { getpid() }
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/ubuntu"))
}

pub fn base_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("JOPE_DISK_STEWARD_BASE") {
        PathBuf::from(p)
    } else {
        home_dir().join("ops/disk_steward")
    }
}

pub fn hostname() -> String {
    if let Ok(s) = fs::read_to_string("/proc/sys/kernel/hostname") {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return trimmed.split('.').next().unwrap_or(trimmed).to_string();
        }
    }
    std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
}

pub fn statvfs_path(path: &Path) -> io::Result<Statvfs> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in path"))?;
    let mut buf = Statvfs::default();
    let rc = unsafe { statvfs(c_path.as_ptr(), &mut buf) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(buf)
}

#[derive(Debug, Clone, Copy)]
pub struct DiskStats {
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub total_capacity_bytes: u64,
    pub used_percent: f64,
}

pub fn disk_usage(path: &Path) -> io::Result<DiskStats> {
    let v = statvfs_path(path)?;
    let frsize = if v.f_frsize > 0 { v.f_frsize } else { v.f_bsize };
    let used = v.f_blocks.saturating_sub(v.f_bfree).saturating_mul(frsize);
    let avail = v.f_bavail.saturating_mul(frsize);
    let total = used.saturating_add(avail);
    let used_percent = if total > 0 {
        100.0 * (used as f64) / (total as f64)
    } else {
        100.0
    };
    Ok(DiskStats {
        used_bytes: used,
        available_bytes: avail,
        total_capacity_bytes: v.f_blocks.saturating_mul(frsize),
        used_percent,
    })
}

pub struct FlockGuard {
    file: File,
}

impl FlockGuard {
    pub fn acquire_nonblocking(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { file })
    }

    pub fn acquire_blocking(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { file })
    }
}

impl Drop for FlockGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

/// Convert Unix epoch seconds and nanoseconds into `(year, month, day, hour, min, sec, micros)`.
/// Uses Howard Hinnant's civil_from_days algorithm.
pub fn epoch_to_ymdhms(secs: i64, nsec: u32) -> (i32, u32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400) as u32;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (y + if m <= 2 { 1 } else { 0 }) as i32;
    (year, m, d, hour, min, sec, nsec / 1000)
}

pub fn epoch_to_utc_iso(secs: i64, nsec: u32) -> String {
    let (y, m, d, hh, mm, ss, us) = epoch_to_ymdhms(secs, nsec);
    if us == 0 {
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}+00:00")
    } else {
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{us:06}+00:00")
    }
}

pub fn now_epoch_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn now_epoch_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

pub fn utc_iso() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    epoch_to_utc_iso(d.as_secs() as i64, d.subsec_nanos())
}

pub fn utc_stamp_compact() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (y, m, day, hh, mm, ss, _) = epoch_to_ymdhms(d.as_secs() as i64, 0);
    format!("{y:04}{m:02}{day:02}T{hh:02}{mm:02}{ss:02}Z")
}

pub fn utc_stamp_micros() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (y, m, day, hh, mm, ss, us) = epoch_to_ymdhms(d.as_secs() as i64, d.subsec_nanos());
    format!("{y:04}{m:02}{day:02}T{hh:02}{mm:02}{ss:02}.{us:06}Z")
}

pub fn atomic_write_str(path: &Path, content: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_os = path.as_os_str().to_os_string();
    tmp_os.push(".new");
    let tmp_path = PathBuf::from(tmp_os);
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(content.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, path)
}

pub fn emit_jsonl_locked(path: &Path, json_line: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let rc = unsafe { flock(f.as_raw_fd(), LOCK_EX) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let res = (|| {
        f.write_all(json_line.as_bytes())?;
        f.write_all(b"\n")?;
        f.flush()?;
        f.sync_all()
    })();
    unsafe {
        let _ = flock(f.as_raw_fd(), LOCK_UN);
    }
    res
}
