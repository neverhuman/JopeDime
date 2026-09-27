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

use jope_records::pyjson::{self, Dict, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DELETE_REFUSAL: &str =
    "Deletion is disabled. Use a logged preservation and independent verification batch.";
const MEASURE_TIMEOUT: Duration = Duration::from_secs(600);
const DF_TIMEOUT: Duration = Duration::from_secs(30);

/// A failed step: the Python exception text and, for a failed subprocess, its stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub message: String,
    pub stderr: Option<String>,
}

impl Failure {
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

pub type Outcome<T> = Result<T, Failure>;

/// An `OSError` as Python prints it: `[Errno N] <strerror>: '<path>'`.
fn os_error(error: &std::io::Error, path: &Path) -> Failure {
    match error.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            Failure::new(format!(
                "[Errno {code}] {strerror}: {}",
                pyjson::str_repr(&path.to_string_lossy())
            ))
        }
        None => Failure::new(format!("{}: {error}", path.display())),
    }
}

/// The UTC clock as `datetime.now(timezone.utc)` prints it: `isoformat()` and the receipt stamp
/// `%Y%m%dT%H%M%S.%fZ`.
pub struct Utc {
    pub iso: String,
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

pub fn utc() -> Utc {
    utc_at(SystemTime::now())
}

/// `os.path.abspath`: absolute and lexically normalised (`..` removes the previous component).
pub fn abspath(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut out = PathBuf::from("/");
    for component in joined.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
            _ => {}
        }
    }
    out
}

/// `Path.resolve()` (not strict): symlinks resolved as far as the path exists.
fn resolve_lenient(path: &Path) -> PathBuf {
    let path = abspath(path);
    let mut existing = path.clone();
    let mut rest: Vec<std::ffi::OsString> = vec![];
    loop {
        if let Ok(resolved) = existing.canonicalize() {
            let mut out = resolved;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (
            existing.file_name().map(ToOwned::to_owned),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return path,
        }
    }
}

fn mtime(meta: &fs::Metadata) -> f64 {
    // Python's st_mtime: seconds plus nanoseconds * 1e-9, as one double.
    meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
}

fn lstat(path: &Path) -> Outcome<fs::Metadata> {
    fs::symlink_metadata(path).map_err(|e| os_error(&e, path))
}

/// Why a run must stop now (a pending SIGTERM/SIGINT in `report`), checked between steps.
pub type Interrupt<'a> = &'a dyn Fn() -> Outcome<()>;

pub fn never_interrupted() -> Outcome<()> {
    Ok(())
}

/// `os.walk(root, followlinks=False, onerror=raise)` after refusing a root with symlink
/// components; `visit` sees each directory, its subdirectory names (symlinks to directories
/// included, as `os.walk` lists them; prune by retaining) and its other entries.
pub fn walk(
    root: &Path,
    interrupt: Interrupt<'_>,
    visit: &mut dyn FnMut(&Path, &mut Vec<String>, &[String]) -> Outcome<()>,
) -> Outcome<()> {
    let root = abspath(root);
    let resolved = root.canonicalize().map_err(|e| os_error(&e, &root))?;
    if resolved != root || !root.is_dir() {
        return Err(Failure::new(format!(
            "root must be an existing directory without symlink components: {}",
            root.display()
        )));
    }
    let mut stack = vec![root];
    while let Some(directory) = stack.pop() {
        interrupt()?;
        let entries = fs::read_dir(&directory).map_err(|e| os_error(&e, &directory))?;
        let (mut dirs, mut files) = (vec![], vec![]);
        for entry in entries {
            let entry = entry.map_err(|e| os_error(&e, &directory))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_dir() {
                dirs.push(name);
            } else {
                files.push(name);
            }
        }
        visit(&directory, &mut dirs, &files)?;
        // Depth first in listing order, as os.walk recurses; directory symlinks are not entered.
        for name in dirs.iter().rev() {
            let child = directory.join(name);
            if !fs::symlink_metadata(&child).is_ok_and(|m| m.file_type().is_symlink()) {
                stack.push(child);
            }
        }
    }
    Ok(())
}

/// Run `command` to completion within `limit`, polling `interrupt`; stdout on success.
fn run_bounded(command: &[&str], limit: Duration, interrupt: Interrupt<'_>) -> Outcome<String> {
    let rendered = format!(
        "[{}]",
        command
            .iter()
            .map(|a| pyjson::str_repr(a))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut child = Command::new(command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| os_error(&e, Path::new(command[0])))?;
    let started = Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| os_error(&e, Path::new(command[0])))?
        {
            let output = child
                .wait_with_output()
                .map_err(|e| os_error(&e, Path::new(command[0])))?;
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            if !status.success() {
                let code = status.code().unwrap_or(-1);
                return Err(Failure {
                    message: format!("Command '{rendered}' returned non-zero exit status {code}."),
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

/// Python `int(text)` for the leading `du` field.
fn py_int(text: &str) -> Outcome<u64> {
    let body = text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
    let digits = body.strip_prefix('+').unwrap_or(body);
    if !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits.bytes().all(|b| b.is_ascii_digit() || b == b'_')
    {
        if let Ok(value) = digits.replace('_', "").parse() {
            return Ok(value);
        }
    }
    Err(Failure::new(format!(
        "invalid literal for int() with base 10: {}",
        pyjson::str_repr(text)
    )))
}

/// Allocated bytes of `root` from `/usr/bin/du -s -B1` (through `sudo -n` when privileged).
pub fn allocated(root: &Path, privileged: bool, interrupt: Interrupt<'_>) -> Outcome<u64> {
    let root_text = root.to_string_lossy();
    let mut command = if privileged {
        vec!["/usr/bin/sudo", "-n", "/usr/bin/du"]
    } else {
        vec!["/usr/bin/du"]
    };
    command.extend(["-s", "-B1", "--", root_text.as_ref()]);
    let stdout = run_bounded(&command, MEASURE_TIMEOUT, interrupt)?;
    py_int(stdout.split('\t').next().unwrap_or(""))
}

/// The scan document of `root`: outermost Cargo candidates older than `hours` (all when `None`).
pub fn scan(
    root: &Path,
    hours: Option<u64>,
    interrupt: Interrupt<'_>,
    measure: &dyn Fn(&Path, bool) -> Outcome<u64>,
) -> Outcome<Value> {
    let root = abspath(root);
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
    let mut total: u64 = 0;
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
        total += bytes;
        let row: Dict = [
            (
                "path",
                Value::from(candidate.to_string_lossy().into_owned()),
            ),
            ("allocated_bytes", Value::from(bytes as i64)),
            ("newest_mtime", Value::Float(newest)),
        ]
        .into_iter()
        .collect();
        rows.push(Value::Dict(row));
    }
    let document: Dict = [
        ("completed_at", Value::from(utc().iso)),
        ("root", Value::from(root.to_string_lossy().into_owned())),
        (
            "stale_hours",
            hours.map_or(Value::Null, |h| Value::from(h as i64)),
        ),
        ("candidates", Value::List(rows)),
        ("candidate_allocated_bytes_sum", Value::from(total as i64)),
        ("sum_may_double_count_hardlinks", Value::from(true)),
        // This object is returned only after every traversal and measurement succeeds.
        ("report_only", Value::from(true)),
        ("complete", Value::from(true)),
    ]
    .into_iter()
    .collect();
    Ok(Value::Dict(document))
}

/// `--stale HOURS`: nonnegative whole hours of at most eight digits.
pub fn whole_hours(value: &str) -> Result<u64, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("--stale requires nonnegative whole hours".into());
    }
    // Bound arithmetic, not the reporting/deletion policy (there is no deletion policy).
    if value.len() > 8 {
        return Err("--stale age is too large".into());
    }
    value
        .parse()
        .map_err(|_| "--stale requires nonnegative whole hours".into())
}

/// What `report` runs: the scan of one approved root and the measurement of one directory.
pub struct Hooks<'a> {
    pub scan: &'a dyn Fn(&str, Interrupt<'_>) -> Outcome<Value>,
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

/// Refuse custody inside `/tmp` or any Git working tree, before anything is written.
pub fn custody_refusal(custody: &Path) -> Option<Failure> {
    let output = resolve_lenient(custody);
    let chain: Vec<&Path> = output.ancestors().collect();
    let in_tmp = chain.iter().any(|p| *p == Path::new("/tmp"));
    let in_tree = chain.iter().any(|p| p.join(".git").exists());
    (in_tmp || in_tree)
        .then(|| Failure::new("report custody must be outside /tmp and Git working trees"))
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
    if let Some(refused) = custody_refusal(custody) {
        return Err(refused);
    }
    let output = resolve_lenient(custody);
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
        (
            "custody_argument",
            Value::from(custody.to_string_lossy().into_owned()),
        ),
        (
            "custody_resolved",
            Value::from(output.to_string_lossy().into_owned()),
        ),
        ("missing_unit_files", Value::List(vec![])),
        (
            "invocation_id",
            std::env::var("INVOCATION_ID").map_or(Value::Null, Value::from),
        ),
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
    println!("{}", receipt_path.display());
    Ok(if result.is_ok() { 0 } else { 1 })
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
    let units =
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".config/systemd/user");
    for path in [
        units.join("jope-build-sweep.service"),
        units.join("jope-build-sweep.timer"),
    ] {
        if path.is_file() {
            paths.push(path);
        } else {
            list_mut(receipt, "missing_unit_files")
                .push(Value::from(path.to_string_lossy().into_owned()));
        }
    }
    for path in &paths {
        let sum = sha256_file(path)?;
        receipt
            .get_mut("checksums")
            .and_then(Value::as_dict_mut)
            .expect("checksums")
            .insert(path.to_string_lossy().into_owned(), Value::from(sum));
    }
    let config_path = config.canonicalize().map_err(|e| os_error(&e, config))?;
    let config_bytes = fs::read(&config_path).map_err(|e| os_error(&e, &config_path))?;
    receipt
        .get_mut("checksums")
        .and_then(Value::as_dict_mut)
        .expect("checksums")
        .insert(
            config_path.to_string_lossy().into_owned(),
            Value::from(sha256_file(&config_path)?),
        );
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
        let path = PathBuf::from(item.get("path").and_then(Value::as_str).expect("checked"));
        let resolved = path.canonicalize().map_err(|e| os_error(&e, &path))?;
        if resolved != path || !resolved.is_dir() {
            return Err(Failure::new(format!(
                "measurement roots must be explicit resolved directories: {}",
                path.display()
            )));
        }
        if seen.iter().any(|prior| {
            resolved == *prior || resolved.starts_with(prior) || prior.starts_with(&resolved)
        }) {
            return Err(Failure::new(format!(
                "overlapping measurement root: {}",
                path.display()
            )));
        }
        seen.push(resolved);
    }
    for item in &measurements {
        let root = PathBuf::from(item.get("path").and_then(Value::as_str).expect("checked"));
        let privileged = item.get("privileged") == Some(&Value::Bool(true));
        let bytes = (hooks.measure)(&root, privileged, interrupt)?;
        let mut row = item.as_dict().expect("checked").clone();
        row.insert("allocated_bytes", Value::from(bytes as i64));
        row.insert("observed_at", Value::from(utc().iso));
        list_mut(receipt, "measurements").push(Value::Dict(row));
        save(receipt)?;
    }
    let output_text = output.to_string_lossy();
    let filesystem = run_bounded(&["df", "-B1", output_text.as_ref()], DF_TIMEOUT, interrupt)?;
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
        assert_eq!(py_int("123").unwrap(), 123);
        assert_eq!(py_int(" 7\n").unwrap(), 7);
        assert_eq!(
            py_int("garbage").unwrap_err().message,
            "invalid literal for int() with base 10: 'garbage'"
        );
    }

    #[test]
    fn abspath_is_lexical() {
        assert_eq!(abspath(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
        assert_eq!(abspath(Path::new("/..")), PathBuf::from("/"));
    }

    #[test]
    fn custody_under_tmp_or_a_working_tree_is_refused() {
        assert!(custody_refusal(Path::new("/tmp")).is_some());
        assert!(custody_refusal(Path::new("/tmp/x/y")).is_some());
        let tree = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(custody_refusal(&tree.join("never-created")).is_some());
    }
}
