use super::*;

pub(super) fn mtime(meta: &fs::Metadata) -> f64 {
    // Python's st_mtime: seconds plus nanoseconds * 1e-9, as one double.
    meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
}

pub(super) fn lstat(path: &Path) -> Outcome<fs::Metadata> {
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
pub(super) fn run_bounded(
    command: &[&OsStr],
    limit: Duration,
    interrupt: Interrupt<'_>,
) -> Outcome<String> {
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
pub(super) fn du_bytes(text: &str) -> Outcome<u64> {
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
