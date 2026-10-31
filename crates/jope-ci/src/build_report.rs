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
pub use jope_records::py::abspath;

fn mtime(meta: &fs::Metadata) -> f64 {
    // Python's st_mtime: seconds plus nanoseconds * 1e-9, as one double.
    meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
}

fn lstat(path: &Path) -> Outcome<fs::Metadata> {
    fs::symlink_metadata(path).map_err(|e| os_error(&e, path))
}

/// Why a run must stop now (a pending SIGTERM/SIGINT in `report`), checked between steps.
pub type Interrupt<'a> = &'a dyn Fn() -> Outcome<()>;

/// Interruption callback that always reports success.
pub fn never_interrupted() -> Outcome<()> {
    Ok(())
}

/// What `walk` calls for each directory: the directory, its subdirectory names (prune by
/// retaining) and its other entries. Names are kept as the bytes the filesystem holds (Python's
/// surrogate-escaped `str`), so a name that is not UTF-8 is reached, measured and joined intact.
pub type Visit<'a> = &'a mut dyn FnMut(&Path, &mut Vec<OsString>, &[OsString]) -> Outcome<()>;

/// `os.walk(root, followlinks=False, onerror=raise)` after refusing a root with symlink
/// components; `visit` sees each directory, its subdirectory names (prune by retaining) and its
/// other entries.
///
/// A subdirectory is an entry whose own type is a directory (`d_type`, else `lstat`), as
/// `os.walk` read it for a directory: so a directory that may be listed but not searched is still
/// entered, and entering its subdirectories fails as `os.walk`'s `scandir` raised (it was once
/// taken for a file, because its `stat` failed, and silently not entered). A symlink is listed with
/// the other entries and never entered: `os.walk` listed a link to a directory among the
/// subdirectories (and one it could not `stat` among the files) but never entered it either, and
/// every caller `lstat`s the names of both lists alike and asks nothing else of a link, so where
/// a link is listed is not observable; nothing is followed through a link, so no error is
/// swallowed. Any error listing or typing an entry refuses.
pub fn walk(root: &Path, interrupt: Interrupt<'_>, visit: Visit<'_>) -> Outcome<()> {
    let root = absolute(root)?;
    let resolved = py::resolve(&root, true).map_err(resolve_failure)?;
    // `root.resolve(strict=True) != root`: pathlib compares the spellings, and `abspath` keeps
    // POSIX's distinct `//` root where `realpath` resolves it to `/`, so the Python refused such a
    // root as it refused one with a symlink component. Both sides are normalised, so their bytes
    // compare (`Path`'s `==` would compare components, which have no `//`). The refusal names the
    // root as Python's stderr printed its `str` (an undecodable byte as `\udcff`, never U+FFFD).
    if resolved.as_os_str() != root.as_os_str() || !root.is_dir() {
        return Err(Failure::new(format!(
            "root must be an existing directory without symlink components: {}",
            py::os_display(root.as_os_str())
        )));
    }
    let mut stack = vec![root];
    while let Some(directory) = stack.pop() {
        interrupt()?;
        let entries = fs::read_dir(&directory).map_err(|e| os_error(&e, &directory))?;
        let (mut dirs, mut files) = (vec![], vec![]);
        for entry in entries {
            let entry = entry.map_err(|e| os_error(&e, &directory))?;
            let name = entry.file_name();
            let kind = entry.file_type().map_err(|e| os_error(&e, &entry.path()))?;
            if kind.is_dir() {
                dirs.push(name);
            } else {
                files.push(name);
            }
        }
        visit(&directory, &mut dirs, &files)?;
        // Depth first in listing order, as os.walk recurses; `os.walk` checked `islink` before
        // entering, so an entry replaced by a symlink since it was listed is not entered, and one
        // that can no longer be inspected refuses.
        for name in dirs.iter().rev() {
            let child = directory.join(name);
            if !lstat(&child)?.file_type().is_symlink() {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// Run `command` to completion within `limit`, polling `interrupt`; stdout on success.
///
/// As `subprocess.run(..., capture_output=True, text=True, check=True, timeout=...)`: both streams
/// are decoded (strict UTF-8, universal newlines; stdout first) before the exit status is looked
/// at, so output that is not UTF-8 fails with Python's `UnicodeDecodeError` text, and a failed
/// command with `CalledProcessError`'s (`returned non-zero exit status N.`, or `died with
/// <Signals.SIGKILL: 9>.`). Arguments are passed as the bytes they are.
fn run_bounded(command: &[&OsStr], limit: Duration, interrupt: Interrupt<'_>) -> Outcome<String> {
    let rendered = format!(
        "[{}]",
        command
            .iter()
            .map(|a| crate::py::os_repr(a))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let program = Path::new(command[0]);
    let mut child = Command::new(command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| os_error(&e, program))?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| os_error(&e, program))? {
            let output = child
                .wait_with_output()
                .map_err(|e| os_error(&e, program))?;
            let stdout = jope_records::py::text_output(&output.stdout).map_err(Failure::new)?;
            let stderr = jope_records::py::text_output(&output.stderr).map_err(Failure::new)?;
            if !status.success() {
                return Err(Failure {
                    message: jope_records::py::called_process_error(
                        &rendered,
                        jope_records::py::returncode(status),
                    ),
                    stderr: Some(stderr),
                });
            }
            return Ok(stdout);
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Failure::new(format!(
                "Command '{rendered}' timed out after {} seconds",
                limit.as_secs()
            )));
        }
        if let Err(stop) = interrupt() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(stop);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `int(text)` of the leading `du` field through the strict number grammar
/// ([`jope_records::py::decimal`]): ASCII whitespace, sign and digits only, a value that fits
/// `i64` and is not negative. More than 4300 digits (leading zeros count) refuse in CPython 3.12's
/// words; anything else outside the grammar (which includes underscores, Unicode digits and
/// Unicode whitespace, all of which Python read, and U+001C..U+001F, which it refused) refuses as
/// Python's `ValueError` worded it.
fn du_bytes(text: &str) -> Outcome<u64> {
    match py::int(text) {
        Ok(value) if value >= 0 => Ok(value.unsigned_abs()),
        Err(py::IntError::Limit(message)) => Err(Failure::new(message)),
        _ => Err(Failure::new(format!(
            "invalid literal for int() with base 10: {}",
            pyjson::str_repr(text)
        ))),
    }
}

/// Allocated bytes of `root` from `/usr/bin/du -s -B1` (through `sudo -n` when privileged).
pub fn allocated(root: &Path, privileged: bool, interrupt: Interrupt<'_>) -> Outcome<u64> {
    let mut command: Vec<&OsStr> = if privileged {
        vec![
            "/usr/bin/sudo".as_ref(),
            "-n".as_ref(),
            "/usr/bin/du".as_ref(),
        ]
    } else {
        vec!["/usr/bin/du".as_ref()]
    };
    command.extend([
        "-s".as_ref(),
        "-B1".as_ref(),
        "--".as_ref(),
        root.as_os_str(),
    ]);
    let stdout = run_bounded(&command, MEASURE_TIMEOUT, interrupt)?;
    du_bytes(stdout.split('\t').next().unwrap_or(""))
}

/// The scan document of `root`: outermost Cargo candidates older than `hours` (all when `None`).
pub fn scan(
    root: &Path,
    hours: Option<u64>,
    interrupt: Interrupt<'_>,
    measure: &dyn Fn(&Path, bool) -> Outcome<u64>,
) -> Outcome<Value> {
    let root = absolute(root)?;
    let mut candidates: std::collections::BTreeSet<PathBuf> = Default::default();
    walk(&root, interrupt, &mut |directory, dirs, files| {
        dirs.retain(|d| d != ".git");
        if files.iter().any(|f| f == ".rustc_info.json")
            && lstat(&directory.join(".rustc_info.json"))?
                .file_type()
                .is_file()
        {
            candidates.insert(directory.to_path_buf());
        }
        let name = directory.file_name().and_then(|n| n.to_str());
        if matches!(name, Some("debug" | "release")) && dirs.iter().any(|d| d == ".fingerprint") {
            let parent = directory
                .parent()
                .expect("a debug/release directory has a parent");
            if (parent == root || (parent.starts_with(&root) && parent != root))
                && lstat(&directory.join(".fingerprint"))?.file_type().is_dir()
            {
                candidates.insert(parent.to_path_buf());
            }
        }
        Ok(())
    })?;
    let cutoff = hours.map(|h| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            - h as f64 * 3600.0
    });
    let mut rows = vec![];
    // `sum(r["allocated_bytes"] for r in rows)`: Python's int has no width, so the total is held
    // exactly (in `u128`, which fewer than 2^64 measurements cannot leave) and written exactly.
    let mut total: u128 = 0;
    for candidate in &candidates {
        if candidate
            .ancestors()
            .skip(1)
            .any(|a| candidates.contains(a))
        {
            continue;
        }
        let mut newest = mtime(&lstat(candidate)?);
        // Measure all contents, including nested .git evidence, after identifying candidates.
        // Visit fresh trees too so traversal failures stay visible.
        walk(candidate, interrupt, &mut |directory, dirs, files| {
            for name in dirs.iter().chain(files) {
                newest = newest.max(mtime(&lstat(&directory.join(name))?));
            }
            Ok(())
        })?;
        if cutoff.is_some_and(|c| newest > c) {
            continue;
        }
        let bytes = measure(candidate, false)?;
        total = total
            .checked_add(u128::from(bytes))
            .ok_or_else(|| Failure::new("the allocated-bytes total leaves 128 bits"))?;
        // `du` prints the path it measured, and its output is decoded strictly, so a measured
        // candidate is UTF-8; any other measurement is held to the same rule.
        let path = candidate.to_str().ok_or_else(|| {
            Failure::new(format!(
                "candidate {} is not UTF-8; its surrogate-escaped name cannot be represented",
                crate::py::os_repr(candidate.as_os_str())
            ))
        })?;
        let row: Dict = [
            ("path", Value::from(path)),
            ("allocated_bytes", exact_int(u128::from(bytes))),
            ("newest_mtime", Value::Float(newest)),
        ]
        .into_iter()
        .collect();
        rows.push(Value::Dict(row));
    }
    // A candidate path is UTF-8 here (`du`, which prints it, was decoded strictly). A root that is
    // not UTF-8 would be printed surrogate-escaped (`\udcff`), which this JSON cannot hold: refused.
    let root_text = root.to_str().ok_or_else(|| {
        Failure::new(format!(
            "scan root {} is not UTF-8; its surrogate-escaped name cannot be represented",
            crate::py::os_repr(root.as_os_str())
        ))
    })?;
    let document: Dict = [
        ("completed_at", Value::from(utc().iso)),
        ("root", Value::from(root_text)),
        (
            "stale_hours",
            hours.map_or(Value::Null, |h| exact_int(u128::from(h))),
        ),
        ("candidates", Value::List(rows)),
        ("candidate_allocated_bytes_sum", exact_int(total)),
        ("sum_may_double_count_hardlinks", Value::from(true)),
        // This object is returned only after every traversal and measurement succeeds.
        ("report_only", Value::from(true)),
        ("complete", Value::from(true)),
    ]
    .into_iter()
    .collect();
    Ok(Value::Dict(document))
}

/// A count as the JSON integer Python wrote: `int` has no width, so one beyond `i64` is written as
/// its decimal literal ([`Value::BigInt`]), never wrapped or rounded.
fn exact_int(n: u128) -> Value {
    i64::try_from(n).map_or_else(|_| Value::BigInt(n.to_string()), Value::Int)
}

/// `--stale HOURS`: nonnegative whole hours of at most eight digits (`value.isascii() and
/// value.isdecimal()`, then `int(value)`, which the strict number grammar reads).
pub fn whole_hours(value: &str) -> Result<u64, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("--stale requires nonnegative whole hours".into());
    }
    // Bound arithmetic, not the reporting/deletion policy (there is no deletion policy).
    if value.len() > 8 {
        return Err("--stale age is too large".into());
    }
    py::int(value)
        .ok()
        .and_then(|hours| u64::try_from(hours).ok())
        .ok_or_else(|| "--stale requires nonnegative whole hours".into())
}

/// What `report` runs: the scan of one approved root and the measurement of one directory.
pub struct Hooks<'a> {
    /// Scan an approved root and return its report document.
    pub scan: &'a dyn Fn(&str, Interrupt<'_>) -> Outcome<Value>,
    /// Measure a directory, optionally using the privileged measurement route.
    pub measure: &'a dyn Fn(&Path, bool, Interrupt<'_>) -> Outcome<u64>,
}

fn dumps(value: &Value) -> String {
    pyjson::dumps(value, false, Some(2)) + "\n"
}

fn sha256_file(path: &Path) -> Outcome<String> {
    let bytes = fs::read(path).map_err(|e| os_error(&e, path))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `output = args.custody.resolve()`, refused inside `/tmp` or any Git working tree before anything
/// is written. The custody is resolved as `Path.resolve()` resolved it ([`py::resolve`]): each
/// symlink before a later `..`, so `outer/link/../receipts` with `link -> guarded/sub` is
/// `guarded/receipts` (not the lexical `outer/receipts`), and a missing component is kept as
/// spelled. `/tmp` is looked for among the output and its parents first, then a `.git` from the
/// output upwards (`Path.exists()`: a missing path is not there, and any other error, such as a
/// directory that may not be searched, refuses in the `OSError`'s words, as Python raised it). A
/// symlink loop refuses as pathlib's `RuntimeError` did.
pub fn custody_output(custody: &Path) -> Outcome<PathBuf> {
    let output = py::resolve(custody, false).map_err(resolve_failure)?;
    let refused = || Failure::new("report custody must be outside /tmp and Git working trees");
    if output.ancestors().any(|p| p == Path::new("/tmp")) {
        return Err(refused());
    }
    for directory in output.ancestors() {
        let marker = directory.join(".git");
        if py::path_exists(&marker).map_err(|e| os_error(&e, &marker))? {
            return Err(refused());
        }
    }
    Ok(output)
}

/// `report --config --custody`: scan the two approved roots, measure the configured roots and
/// keep every result and diagnostic under a fresh receipt directory. Returns the exit status
/// and prints the receipt path. `tools` is where the wrapper scripts live (their checksums are
/// part of the receipt, with this executable's and the installed units'); `approved` is the
/// wrapper's `--approved-root` list.
pub fn report(
    config: &Path,
    custody: &Path,
    tools: &Path,
    approved: &[String],
    hooks: &Hooks<'_>,
    interrupt: Interrupt<'_>,
) -> Outcome<i32> {
    let output = custody_output(custody)?;
    // The resolved custody is recorded (`custody_resolved`) and the receipt path under it printed
    // (`print(receipt_path)`, after everything was written). One that is not UTF-8 refuses before
    // anything is created, with the `UnicodeEncodeError` Python's print raised under a strict UTF-8
    // stdout (`en_US.UTF-8`; under C and `C.UTF-8` it printed the raw bytes: stricter there, and
    // stricter in writing nothing where the Python had written the receipt).
    let output_text = output.to_str().ok_or_else(|| {
        Failure::new(
            py::utf8_encode_error(output.as_os_str())
                .expect("a name that is not UTF-8 has an undecodable byte"),
        )
    })?;
    // `str(args.custody)` is recorded (`custody_argument`): one that is not UTF-8 refuses too
    // (stricter; the command line refuses it first).
    let argument = custody_argument(custody)?;
    // `os.environ.get("INVOCATION_ID")` into the receipt: a value that is not UTF-8 (which Python
    // held surrogate-escaped) cannot be written by this port's JSON, so it refuses before anything
    // is created (stricter); it was never read as absent.
    let invocation_id = match std::env::var("INVOCATION_ID") {
        Ok(id) => Value::from(id),
        Err(std::env::VarError::NotPresent) => Value::Null,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Failure::new(
                "INVOCATION_ID is not UTF-8; its surrogate-escaped value cannot be represented",
            ))
        }
    };
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&output)
        .map_err(|e| os_error(&e, &output))?;
    let run = output.join(utc().stamp);
    fs::DirBuilder::new()
        .mode(0o700)
        .create(&run)
        .map_err(|e| os_error(&e, &run))?;
    let receipt_path = run.join("receipt.json");
    let mut receipt: Dict = [
        ("started_at", Value::from(utc().iso)),
        ("status", Value::from("running")),
        ("exit_status", Value::Null),
        ("checksums", Value::Dict(Dict::new())),
        ("measurements", Value::List(vec![])),
        ("scan_outputs", Value::List(vec![])),
        ("custody_argument", Value::from(argument)),
        ("custody_resolved", Value::from(output_text)),
        ("missing_unit_files", Value::List(vec![])),
        ("invocation_id", invocation_id),
    ]
    .into_iter()
    .collect();
    let save = |receipt: &Dict| -> Outcome<()> {
        fs::write(&receipt_path, dumps(&Value::Dict(receipt.clone())))
            .map_err(|e| os_error(&e, &receipt_path))
    };
    save(&receipt)?;
    let result = run_report(
        config,
        &output,
        &run,
        tools,
        approved,
        hooks,
        interrupt,
        &mut receipt,
        &save,
    );
    if let Err(error) = &result {
        receipt.insert("status", Value::from("failed"));
        receipt.insert("exit_status", Value::from(1i64));
        receipt.insert("error", Value::from(error.message.clone()));
        let mut text = format!("{}\n", error.message);
        if let Some(stderr) = error.stderr.as_deref().filter(|s| !s.is_empty()) {
            text.push_str(stderr);
        }
        let _ = fs::write(run.join("failure.txt"), text);
    }
    receipt.insert("completed_at", Value::from(utc().iso));
    save(&receipt)?;
    println!(
        "{}",
        receipt_path
            .to_str()
            .expect("a UTF-8 custody, an ASCII stamp and file name")
    );
    Ok(if result.is_ok() { 0 } else { 1 })
}

/// A measurement root: `path = Path(text)`, `resolved = path.resolve(strict=True)`, refused unless
/// `resolved == path` and it is a directory. pathlib compares the spellings after parsing, so a
/// trailing separator or `.` is the same path, while `..` or POSIX's distinct `//` root (which
/// `resolve` turns into `/`) is not. Returns the path as `str()` prints it and the resolved path.
fn explicit_root(text: &str) -> Outcome<(PosixPath, PathBuf)> {
    let spelled = PosixPath::parse(text);
    let resolved = py::resolve(Path::new(text), true).map_err(resolve_failure)?;
    let same = resolved
        .to_str()
        .is_some_and(|r| PosixPath::parse(r) == spelled);
    if !same || !resolved.is_dir() {
        return Err(Failure::new(format!(
            "measurement roots must be explicit resolved directories: {spelled}"
        )));
    }
    Ok((spelled, resolved))
}

/// `Path.home() / ".config/systemd/user"` (Python 3.12: `expanduser("~")` is `$HOME` without its
/// trailing separators), spelled as pathlib spells it (repeated separators and `.` dropped,
/// POSIX's `//` root kept). `HOME` is read only when it is set, UTF-8 and absolute:
///
/// - one that begins with `~` (`~`, `~/x`, `~user/x`, whatever follows) is refused with the
///   `RuntimeError` Python 3.12's `Path.home()` raised, `Could not determine home directory.`, at
///   the same step, so the receipt fails as the Python's did;
/// - unset (Python read the password database), not UTF-8 (Python held it surrogate-escaped),
///   empty (Python took `/`) or relative (Python located the units from the working directory):
///   refused, stricter. None is ever read as an empty, default or relative directory.
fn units_dir() -> Outcome<PosixPath> {
    use std::os::unix::ffi::OsStrExt;
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Failure::new(
            "HOME is not set; the unit files under it cannot be located",
        ));
    };
    if home.as_bytes().starts_with(b"~") {
        return Err(Failure::new("Could not determine home directory."));
    }
    let Some(home) = home.to_str() else {
        return Err(Failure::new(
            "HOME is not UTF-8; its surrogate-escaped value cannot be represented",
        ));
    };
    if !home.starts_with('/') {
        return Err(Failure::new(format!(
            "HOME {} is not an absolute path; the unit files under it cannot be located",
            pyjson::str_repr(home)
        )));
    }
    let trimmed = home.trim_end_matches('/');
    let home = if trimmed.is_empty() { "/" } else { trimmed };
    Ok(PosixPath::parse(home).join(".config/systemd/user"))
}

/// `str(args.custody)` (`--custody` is parsed with `type=Path`): the argument as pathlib spells it.
/// One that is not UTF-8 refuses (see [`receipt_text`]).
fn custody_argument(custody: &Path) -> Outcome<String> {
    Ok(PosixPath::parse(receipt_text(custody, "custody argument")?).to_string())
}

fn list_mut<'a>(receipt: &'a mut Dict, key: &str) -> &'a mut Vec<Value> {
    receipt
        .get_mut(key)
        .and_then(Value::as_list_mut)
        .expect("receipt list")
}

#[allow(clippy::too_many_arguments)]
fn run_report(
    config: &Path,
    output: &Path,
    run: &Path,
    tools: &Path,
    approved: &[String],
    hooks: &Hooks<'_>,
    interrupt: Interrupt<'_>,
    receipt: &mut Dict,
    save: &dyn Fn(&Dict) -> Outcome<()>,
) -> Outcome<()> {
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|e| os_error(&e, Path::new("jope-ci")))?;
    let mut paths = vec![
        exe,
        tools.join("cargo-build-dirs.sh"),
        tools.join("jope-build-report.sh"),
    ];
    let units = units_dir()?;
    for name in ["jope-build-sweep.service", "jope-build-sweep.timer"] {
        // `path.is_file()`: a missing unit is recorded as missing; any other error (a unit
        // directory that may not be searched) fails the run, as the Python raised it.
        let spelled = units.join(name).to_string();
        let path = PathBuf::from(&spelled);
        if py::path_is_file(&path).map_err(|e| os_error(&e, &path))? {
            paths.push(path);
        } else {
            list_mut(receipt, "missing_unit_files").push(Value::from(spelled));
        }
    }
    // `receipt["checksums"][str(path)] = sha256(path.read_bytes())`: the file is read first, then
    // its name is the key. A name that is not UTF-8 (this executable, the tools directory or the
    // configuration reached through a directory named with such a byte) was a surrogate-escaped
    // key in the Python's receipt; here it fails the run (stricter), never a lossy key.
    for path in &paths {
        let sum = sha256_file(path)?;
        let key = receipt_text(path, "checksum key")?.to_owned();
        receipt
            .get_mut("checksums")
            .and_then(Value::as_dict_mut)
            .expect("checksums")
            .insert(key, Value::from(sum));
    }
    let config_path = config.canonicalize().map_err(|e| os_error(&e, config))?;
    let config_bytes = fs::read(&config_path).map_err(|e| os_error(&e, &config_path))?;
    let sum = sha256_file(&config_path)?;
    let key = receipt_text(&config_path, "checksum key")?.to_owned();
    receipt
        .get_mut("checksums")
        .and_then(Value::as_dict_mut)
        .expect("checksums")
        .insert(key, Value::from(sum));
    let config = pyjson::parse_bytes(&config_bytes).map_err(Failure::new)?;
    let keys_present = config.as_dict().is_some_and(|d| {
        ["cargo_roots", "stale_hours", "measurement_roots"]
            .iter()
            .all(|k| d.contains_key(k))
    });
    if !keys_present {
        return Err(Failure::new(
            "config requires cargo_roots, stale_hours and measurement_roots",
        ));
    }
    receipt.insert("config", config.clone());
    let approved_list = Value::List(approved.iter().map(|r| Value::from(r.as_str())).collect());
    let roots = config.get("cargo_roots").expect("checked");
    if approved.len() != 2
        || roots != &approved_list
        || config.get("stale_hours") != Some(&Value::Int(48))
    {
        return Err(Failure::new(
            "service requires the two approved Cargo roots and a 48-hour filter",
        ));
    }
    let measurements = config
        .get("measurement_roots")
        .and_then(Value::as_list)
        .ok_or_else(|| Failure::new("measurement_roots must be a list"))?
        .clone();
    for (index, root) in approved.iter().map(String::as_str).enumerate() {
        let result_name = format!("candidates-{index}.json");
        let stderr_name = format!("scan-{index}.stderr");
        let row: Dict = [
            ("root", Value::from(root)),
            ("output", Value::from(result_name.clone())),
            ("stderr", Value::from(stderr_name.clone())),
        ]
        .into_iter()
        .collect();
        list_mut(receipt, "scan_outputs").push(Value::Dict(row));
        let diagnostic = run.join(&stderr_name);
        fs::write(&diagnostic, "").map_err(|e| os_error(&e, &diagnostic))?;
        match (hooks.scan)(root, interrupt) {
            Ok(document) => {
                let path = run.join(&result_name);
                fs::write(&path, dumps(&document)).map_err(|e| os_error(&e, &path))?;
            }
            Err(error) => {
                let text = format!(
                    "{}\n{}",
                    error.message,
                    error.stderr.clone().unwrap_or_default()
                );
                let _ = fs::write(&diagnostic, text);
                return Err(error);
            }
        }
    }
    // Keep Cargo reports even if a separate comparison root is stale: validate every root
    // before any measurement.
    let mut seen: Vec<PathBuf> = vec![];
    for item in &measurements {
        let valid = item.as_dict().is_some_and(|d| {
            d.get("path").and_then(Value::as_str).is_some()
                && d.get("group").and_then(Value::as_str).is_some()
                && matches!(d.get("privileged"), None | Some(Value::Bool(_)))
        });
        if !valid {
            return Err(Failure::new(
                "each measurement requires string path/group and optional boolean privileged",
            ));
        }
        let (path, resolved) =
            explicit_root(item.get("path").and_then(Value::as_str).expect("checked"))?;
        if seen.iter().any(|prior| {
            resolved == *prior || resolved.starts_with(prior) || prior.starts_with(&resolved)
        }) {
            return Err(Failure::new(format!(
                "overlapping measurement root: {path}"
            )));
        }
        seen.push(resolved);
    }
    for item in &measurements {
        let root = PathBuf::from(item.get("path").and_then(Value::as_str).expect("checked"));
        let privileged = item.get("privileged") == Some(&Value::Bool(true));
        let bytes = (hooks.measure)(&root, privileged, interrupt)?;
        let mut row = item.as_dict().expect("checked").clone();
        row.insert("allocated_bytes", exact_int(u128::from(bytes)));
        row.insert("observed_at", Value::from(utc().iso));
        list_mut(receipt, "measurements").push(Value::Dict(row));
        save(receipt)?;
    }
    let filesystem = run_bounded(
        &["df".as_ref(), "-B1".as_ref(), output.as_os_str()],
        DF_TIMEOUT,
        interrupt,
    )?;
    let path = run.join("filesystem.txt");
    fs::write(&path, filesystem).map_err(|e| os_error(&e, &path))?;
    receipt.insert("status", Value::from("success"));
    receipt.insert("exit_status", Value::from(0i64));
    Ok(())
}

/// SIGTERM/SIGINT become a failed receipt ("report interrupted by signal N") instead of a
/// `running` one.
pub fn signal_interrupt() -> std::io::Result<impl Fn() -> Outcome<()>> {
    let flag = Arc::new(AtomicUsize::new(0));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register_usize(signal, Arc::clone(&flag), signal as usize)?;
    }
    Ok(move || match flag.load(Ordering::SeqCst) {
        0 => Ok(()),
        signal => Err(Failure::new(format!(
            "report interrupted by signal {signal}"
        ))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_match_python_isoformat_and_the_receipt_stamp() {
        let t = UNIX_EPOCH + Duration::new(1_790_000_000, 123_456_789);
        let u = utc_at(t);
        assert_eq!(u.iso, "2026-09-21T14:13:20.123456+00:00");
        assert_eq!(u.stamp, "20260921T141320.123456Z");
        let whole = utc_at(UNIX_EPOCH + Duration::from_secs(951_782_400));
        assert_eq!(whole.iso, "2000-02-29T00:00:00+00:00");
        assert_eq!(whole.stamp, "20000229T000000.000000Z");
    }

    #[test]
    fn stale_hours_and_du_parsing_follow_the_python_rules() {
        assert_eq!(whole_hours("48"), Ok(48));
        assert!(whole_hours("-1").is_err());
        assert!(whole_hours("wat").is_err());
        assert!(whole_hours("").is_err());
        assert_eq!(
            whole_hours("123456789"),
            Err("--stale age is too large".into())
        );
        assert_eq!(whole_hours("00000048"), Ok(48));
        assert_eq!(du_bytes("123").unwrap(), 123);
        assert_eq!(du_bytes(" 7\n").unwrap(), 7);
        assert_eq!(du_bytes("+7").unwrap(), 7);
        assert_eq!(
            du_bytes("garbage").unwrap_err().message,
            "invalid literal for int() with base 10: 'garbage'"
        );
        // CPython's limit of 4300 digits applies to a decimal integer before it is converted:
        // leading zeros count, the sign and the whitespace do not.
        let limit = "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit";
        let zeros = |n: usize| "0".repeat(n);
        assert_eq!(du_bytes(&format!("{}45\n", zeros(4298))).unwrap(), 45);
        assert_eq!(du_bytes(&format!("+{}45", zeros(4298))).unwrap(), 45);
        for text in [
            format!("{}45", zeros(4299)),
            format!(" +{}45\n", zeros(4299)),
        ] {
            assert_eq!(du_bytes(&text).unwrap_err().message, limit);
        }
        // Outside the strict grammar: what Python refused (U+001C..U+001F, which `str.strip()`
        // removes but `int()` did not) and what it read (underscores, Unicode digits and
        // whitespace, stricter), a negative size, and a size beyond `i64`.
        for text in [
            "\u{1f}45",
            "45\u{1c}",
            "4_5",
            &format!("{}45", "0_".repeat(4299)),
            "\u{664}\u{665}",
            "\u{a0}45",
            "-1",
            "9223372036854775808",
        ] {
            assert_eq!(
                du_bytes(text).unwrap_err().message,
                format!(
                    "invalid literal for int() with base 10: {}",
                    pyjson::str_repr(text)
                ),
                "{text:?}"
            );
        }
    }

    /// `sum(r["allocated_bytes"] for r in rows)` is exact at any size (review round eight): three
    /// measurements of `2^63 - 1` bytes total `3 * (2^63 - 1)`, written as that integer, which a
    /// 64-bit total wrapped (a release build) or panicked on (a debug build). Not captured from
    /// the original: its measurements come from `/usr/bin/du`, which prints no such size here.
    #[test]
    fn candidate_bytes_sum_exactly_past_64_bits() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        for name in ["a", "b", "c"] {
            std::fs::create_dir(root.join(name)).unwrap();
            std::fs::write(root.join(name).join(".rustc_info.json"), "{}").unwrap();
        }
        let measured = i64::MAX.unsigned_abs();
        let document = scan(&root, None, &never_interrupted, &|_, _| Ok(measured)).unwrap();
        let rows = document.get("candidates").and_then(Value::as_list).unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row.get("allocated_bytes"), Some(&Value::Int(i64::MAX)));
        }
        let total = document.get("candidate_allocated_bytes_sum").unwrap();
        assert!(matches!(total, Value::BigInt(_)), "{total:?}");
        assert_eq!(total.repr(), "27670116110564327421");
        assert!(
            dumps(&document).contains("\"candidate_allocated_bytes_sum\": 27670116110564327421,")
        );
    }

    /// The libtest name of a test in this module (its path without the crate).
    fn test_name(name: &str) -> String {
        let module = module_path!().split_once("::").expect("a crate path").1;
        format!("{module}::{name}")
    }

    #[test]
    fn abspath_is_lexical() {
        let abs = |p: &str| abspath(Path::new(p)).unwrap();
        assert_eq!(abs("/a/b/../c/./d"), PathBuf::from("/a/c/d"));
        assert_eq!(abs("/.."), PathBuf::from("/"));
        // `posixpath.normpath` keeps exactly two leading slashes; `Path`'s `==` would not see them.
        let spelled = |p: &str| abs(p).into_os_string();
        assert_eq!(spelled("//a/./b//c/"), "//a/b/c");
        assert_eq!(spelled("//a/.."), "//");
        assert_eq!(spelled("///a/b"), "/a/b");
        assert_eq!(spelled("////"), "/");
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(abs("x/../y"), cwd.join("y"));
    }

    /// A relative scan root or custody from a working directory that no longer exists refuses as
    /// `os.getcwd()` raised; it is never taken relative to `/`. Run in a child process, whose
    /// working directory is removed under it, so this test's own process keeps its directory.
    #[test]
    fn a_deleted_working_directory_refuses_relative_paths() {
        if std::env::var_os("JOPE_CI_DELETED_CWD_CHILD").is_some() {
            let failure = |r: Outcome<Value>| r.map(|_| ()).unwrap_err().message;
            let scanned = failure(scan(Path::new("x"), None, &never_interrupted, &|_, _| {
                Ok(0)
            }));
            let custody = custody_output(Path::new("receipts")).unwrap_err().message;
            println!("scan={scanned}\ncustody={custody}");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let gone = temp.path().join("gone");
        std::fs::create_dir(&gone).unwrap();
        let exe = std::env::current_exe().unwrap();
        let script = format!(
            "cd '{}' && rmdir '{}' && exec '{}' --exact --nocapture {}",
            gone.display(),
            gone.display(),
            exe.display(),
            test_name("a_deleted_working_directory_refuses_relative_paths")
        );
        let output = std::process::Command::new("/bin/sh")
            .args(["-c", &script])
            .env("JOPE_CI_DELETED_CWD_CHILD", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{stdout}");
        assert!(
            stdout.contains("scan=[Errno 2] No such file or directory\n"),
            "{stdout}"
        );
        assert!(
            stdout.contains("custody=[Errno 2] No such file or directory\n"),
            "{stdout}"
        );
    }

    /// `Path.home() / ".config/systemd/user"` as pathlib spells it. A `HOME` beginning with `~`
    /// refuses with the `RuntimeError` text Python 3.12 raised; an unset, undecodable, empty or
    /// relative one refuses (stricter; review round six). Checked in a child process, which owns
    /// its environment.
    #[test]
    fn unit_files_are_located_as_path_home_located_them() {
        if let Some(case) = std::env::var_os("JOPE_CI_UNITS_CHILD") {
            let located = units_dir().map(|p| p.to_string()).map_err(|e| e.message);
            println!("{}={located:?}", case.to_string_lossy());
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let run = |case: &str, home: Option<&std::ffi::OsStr>| {
            let mut command = std::process::Command::new(&exe);
            command
                .args(["--exact", "--nocapture"])
                .arg(test_name(
                    "unit_files_are_located_as_path_home_located_them",
                ))
                .env("JOPE_CI_UNITS_CHILD", case);
            match home {
                Some(home) => command.env("HOME", home),
                None => command.env_remove("HOME"),
            };
            String::from_utf8_lossy(&command.output().unwrap().stdout).into_owned()
        };
        use std::os::unix::ffi::OsStrExt;
        for (case, home, want) in [
            ("plain", Some("/h/u"), r#"Ok("/h/u/.config/systemd/user")"#),
            (
                "trailing",
                Some("/h/u//"),
                r#"Ok("/h/u/.config/systemd/user")"#,
            ),
            (
                "inner",
                Some("/h//./u"),
                r#"Ok("/h/u/.config/systemd/user")"#,
            ),
            (
                "double-root",
                Some("//h"),
                r#"Ok("//h/.config/systemd/user")"#,
            ),
            ("root", Some("/"), r#"Ok("/.config/systemd/user")"#),
            (
                "slash-tilde",
                Some("/~x"),
                r#"Ok("/~x/.config/systemd/user")"#,
            ),
            ("unset", None, "Err(\"HOME is not set"),
            (
                "tilde",
                Some("~"),
                r#"Err("Could not determine home directory.")"#,
            ),
            (
                "tilde-path",
                Some("~/x"),
                r#"Err("Could not determine home directory.")"#,
            ),
            (
                "tilde-user",
                Some("~nobody/x"),
                r#"Err("Could not determine home directory.")"#,
            ),
            ("empty", Some(""), "Err(\"HOME '' is not an absolute path"),
            (
                "relative",
                Some("rel/home"),
                "Err(\"HOME 'rel/home' is not an absolute path",
            ),
        ] {
            let out = run(case, home.map(std::ffi::OsStr::new));
            assert!(out.contains(&format!("{case}={want}")), "{case}: {out}");
        }
        let out = run("bytes", Some(std::ffi::OsStr::from_bytes(b"/h/\xff")));
        assert!(out.contains("bytes=Err(\"HOME is not UTF-8"), "{out}");
        // `~` first: `expanduser` returned it unexpanded whatever followed.
        let out = run("tilde-bytes", Some(std::ffi::OsStr::from_bytes(b"~\xff")));
        assert!(
            out.contains(r#"tilde-bytes=Err("Could not determine home directory.")"#),
            "{out}"
        );
    }

    /// A root spelled with POSIX's distinct `//` root: `resolve(strict=True)` spells it `/`, so the
    /// Python refused it as a scan root and as a measurement root; `///`, `/./` and a trailing
    /// separator spell the same path and were accepted.
    #[test]
    fn roots_spelled_with_a_double_slash_are_refused_as_python_refused_them() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().canonicalize().unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        let text = dir.to_str().unwrap();
        let walk_root = |root: String| {
            walk(Path::new(&root), &never_interrupted, &mut |_, _, _| Ok(())).map_err(|e| e.message)
        };
        let refusal = |root: &str| {
            format!("root must be an existing directory without symlink components: {root}")
        };
        assert_eq!(
            walk_root(format!("/{text}")),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(
            walk_root(format!("/{text}/sub/..")),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(walk_root(format!("//{text}")), Ok(()));
        assert_eq!(walk_root(format!("{text}/./")), Ok(()));
        let measured = |root: String| explicit_root(&root).map(|(p, _)| p.to_string());
        let refusal =
            |root: &str| format!("measurement roots must be explicit resolved directories: {root}");
        assert_eq!(
            measured(format!("/{text}")).map_err(|e| e.message),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(
            measured(format!("{text}/sub/..")).map_err(|e| e.message),
            Err(refusal(&format!("{text}/sub/..")))
        );
        assert_eq!(measured(format!("//{text}")).unwrap(), text);
        assert_eq!(measured(format!("{text}/./")).unwrap(), text);
        assert_eq!(custody_argument(Path::new("//c//d/./")).unwrap(), "//c/d");
        assert_eq!(custody_argument(Path::new("c/")).unwrap(), "c");
    }

    #[test]
    fn custody_under_tmp_or_a_working_tree_is_refused() {
        let refusal = |custody: &Path| custody_output(custody).map_err(|e| e.message);
        let refused = Err("report custody must be outside /tmp and Git working trees".to_owned());
        assert_eq!(refusal(Path::new("/tmp")), refused);
        assert_eq!(refusal(Path::new("/tmp/x/y")), refused);
        assert_eq!(refusal(Path::new("//tmp/never-created")), refused);
        // Build archives used by remote checks have no enclosing .git marker. Make the worktree
        // marker explicit so this test checks custody policy independently of the checkout type.
        let tree = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        std::fs::write(tree.path().join(".git"), b"gitdir: /not-needed\n").unwrap();
        assert_eq!(refusal(&tree.path().join("never-created")), refused);
        assert_eq!(
            refusal(&tree.path().join("src/never/../../never-created")),
            refused
        );
    }

    /// Round four of the review of PR #117: custody is resolved as `Path.resolve()` resolved it,
    /// each symlink before a later `..`. With `outer/link -> guarded/sub` and a `.git` in
    /// `guarded`, `outer/link/../receipts` is `guarded/receipts` and is refused (the lexical
    /// `outer/receipts` was accepted before); the plain custody and a `..` after a directory that
    /// is not a link resolve where they did. Nothing is created. The tree is under the build's
    /// temporary directory, which must itself be outside `/tmp` and any working tree for the
    /// accepted cases to mean anything, so they are checked only there.
    #[test]
    fn custody_is_resolved_symlinks_first() {
        let temp = tempfile::tempdir().unwrap();
        let c = temp.path().canonicalize().unwrap();
        for dir in ["outer/realdir", "guarded/sub", "guarded/.git"] {
            std::fs::create_dir_all(c.join(dir)).unwrap();
        }
        std::os::unix::fs::symlink(c.join("guarded/sub"), c.join("outer/link")).unwrap();
        std::os::unix::fs::symlink("loop", c.join("outer/loop")).unwrap();
        let refused = Err("report custody must be outside /tmp and Git working trees".to_owned());
        let custody = |spelled: &str| {
            custody_output(&PathBuf::from(format!("{}/{spelled}", c.display())))
                .map_err(|e| e.message)
        };
        for into_guarded in [
            "outer/link/../receipts",
            "outer/link/../sub/../receipts",
            "outer/missing/../link/../receipts",
            "outer/link/receipts",
            "outer/link/..",
        ] {
            assert_eq!(custody(into_guarded), refused, "{into_guarded}");
        }
        assert_eq!(
            custody("outer/loop/x"),
            Err(format!(
                "Symlink loop from {}",
                crate::py::repr(&format!("{}/outer/loop/x", c.display()))
            ))
        );
        let outside = custody_output(&c).is_ok();
        if outside {
            for (spelled, resolved) in [
                ("outer/receipts", "outer/receipts"),
                ("outer/realdir/../receipts", "outer/receipts"),
                ("outer/missing/../receipts", "outer/receipts"),
                ("outer/link/../../outer/receipts", "outer/receipts"),
                ("outer/loop/../receipts", "outer/receipts"),
            ] {
                assert_eq!(custody(spelled), Ok(c.join(resolved)), "{spelled}");
            }
        }
        assert!(!c.join("outer/receipts").exists() && !c.join("guarded/receipts").exists());
    }
}
