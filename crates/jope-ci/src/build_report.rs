//! `jope-ci build-report scan|report`: report Cargo build-directory candidates and allocated disk
//! usage without changing scanned files.
//!
//! Rust port of `deploy/maintenance/jope_report.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`): the same candidate rule, JSON documents,
//! receipts and failure behaviour. A candidate is a directory holding a regular
//! `.rustc_info.json`, or the parent of a `debug`/`release` directory holding a real
//! `.fingerprint` directory, inside the scanned root. Only outermost candidates are reported;
//! their age is the newest modification time of anything beneath them. Directory symlinks are
//! never followed, `.git` is not searched for markers, and every traversal or measurement error
//! fails the run visibly. Allocated bytes come from `/usr/bin/du -s -B1` (optionally through
//! `/usr/bin/sudo -n`). `--delete` is refused before anything is read.
//!
//! The two approved Cargo roots are host layout, not product code (ADR-0003, `just lint-paths`).
//! The Python kept them as a constant in `deploy/maintenance/`; they now stay in that directory,
//! in the installed wrapper `deploy/maintenance/jope-build-report.sh`, which passes them with
//! `--approved-root` and whose checksum every receipt records. `report` fails unless exactly two
//! are given and the configuration names exactly those.

pub use jope_records::py::abspath;
use jope_records::py::{self, PosixPath, ResolveError};
use jope_records::pyjson::{self, Dict, Value};
use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Stable refusal message used when callers request deletion from a report operation.
pub const DELETE_REFUSAL: &str =
    "Deletion is disabled. Use a logged preservation and independent verification batch.";
const MEASURE_TIMEOUT: Duration = Duration::from_secs(600);
const DF_TIMEOUT: Duration = Duration::from_secs(30);

/// A failed step: the Python exception text and, for a failed subprocess, its stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Python-compatible diagnostic for the failed step.
    pub message: String,
    /// Captured standard error when a subprocess failed.
    pub stderr: Option<String>,
}

impl Failure {
    /// Create a failed outcome with a diagnostic and no captured subprocess stderr.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            stderr: None,
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Result type used by build-report operations, carrying their compatibility diagnostic.
pub type Outcome<T> = Result<T, Failure>;

/// An `OSError` as Python prints it: `[Errno N] <strerror>: '<path>'`, a path that is not UTF-8
/// shown as Python's surrogate-escaped `repr` (`'\udcff'`), never converted lossily.
fn os_error(error: &std::io::Error, path: &Path) -> Failure {
    Failure::new(py::os_error_path(error, path))
}

/// A path the receipt records or the command prints, as text. One that is not UTF-8 was a
/// surrogate-escaped `str` in Python, which this port's JSON and output cannot hold; it refuses
/// (`what` names the field) instead of being converted lossily (U+FFFD names another path).
fn receipt_text<'a>(path: &'a Path, what: &str) -> Outcome<&'a str> {
    path.to_str().ok_or_else(|| {
        Failure::new(format!(
            "{what} {} is not UTF-8; its surrogate-escaped name cannot be represented",
            crate::py::os_repr(path.as_os_str())
        ))
    })
}

/// `Path.resolve()`'s failure as Python printed it: the `OSError` naming the path up to the
/// component it could not resolve (`[Errno 2] ...: '/nonexistent-q'`), or pathlib's
/// `RuntimeError` for a symlink loop.
fn resolve_failure(error: ResolveError) -> Failure {
    match error {
        ResolveError::Os { error, path } => os_error(&error, &path),
        ResolveError::Loop { path } => Failure::new(format!(
            "Symlink loop from {}",
            crate::py::os_repr(path.as_os_str())
        )),
        ResolveError::Cwd(error) => Failure::new(py::os_error_bare(&error)),
    }
}

/// `os.path.abspath(path)`: a relative path from a working directory that cannot be read (it was
/// deleted) refuses as `os.getcwd()` raised (`[Errno 2] No such file or directory`), instead of
/// being taken relative to `/`.
fn absolute(path: &Path) -> Outcome<PathBuf> {
    abspath(path).map_err(|e| Failure::new(py::os_error_bare(&e)))
}

/// The UTC clock as `datetime.now(timezone.utc)` prints it: `isoformat()` and the receipt stamp
/// `%Y%m%dT%H%M%S.%fZ`.
pub struct Utc {
    /// ISO 8601 UTC date and time.
    pub iso: String,
    /// Compact UTC timestamp used in receipt directory names.
    pub stamp: String,
}

/// Civil date of a day count since 1970-01-01 (proleptic Gregorian).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe as i64 + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Format a system time as ISO 8601 UTC text and the compact receipt timestamp.
pub fn utc_at(time: SystemTime) -> Utc {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs() as i64;
    let micros = since.subsec_micros();
    let (year, month, day) = civil(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    let fraction = if micros == 0 {
        String::new()
    } else {
        format!(".{micros:06}")
    };
    Utc {
        iso: format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}{fraction}+00:00"),
        stamp: format!("{year:04}{month:02}{day:02}T{h:02}{m:02}{s:02}.{micros:06}Z"),
    }
}

/// Format the current system time as UTC text and a receipt timestamp.
pub fn utc() -> Utc {
    utc_at(SystemTime::now())
}

/// `os.path.abspath`, lexical: [`jope_records::py::abspath`]. The Python applied it to a scan root
/// before resolving it, so there (and only there) `..` is removed before symlinks are resolved.
mod filesystem;
pub use filesystem::allocated;
pub use filesystem::never_interrupted;
pub use filesystem::walk;
pub use filesystem::Interrupt;
pub use filesystem::Visit;
pub(super) use filesystem::*;
mod scan;
pub use scan::scan;
pub use scan::whole_hours;
pub(super) use scan::*;
mod reporting;
pub use reporting::custody_output;
pub use reporting::report;
pub use reporting::Hooks;
pub(super) use reporting::*;
mod runner;
pub use runner::signal_interrupt;
pub(super) use runner::*;

#[cfg(test)]
#[path = "build_report/tests.rs"]
mod tests;
