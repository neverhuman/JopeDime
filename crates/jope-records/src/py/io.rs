use super::*;

/// `Path.read_bytes()`.
pub fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::io(path, e))
}

/// `Path.read_text()`: strict UTF-8 (a bad byte refuses with CPython's `UnicodeDecodeError`
/// message) with universal newlines (`\r\n` and `\r` become `\n`).
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = read_bytes(path)?;
    let text = pyjson::decode_utf8(&bytes).map_err(|message| Error::parse(path, message))?;
    Ok(universal_newlines(text))
}

/// `Path.read_text(errors="replace")`: each invalid UTF-8 sequence becomes U+FFFD (the same
/// maximal subparts CPython replaces), with universal newlines.
pub fn read_text_replace(path: &Path) -> Result<String> {
    Ok(universal_newlines(&String::from_utf8_lossy(&read_bytes(
        path,
    )?)))
}

/// Text-mode newline translation (`newline=None`): `\r\n` and a lone `\r` become `\n`.
pub fn universal_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// A subprocess stream read with `text=True`: strict UTF-8 with universal newlines, or CPython's
/// `UnicodeDecodeError` message (Python raised it, and did not return the output, even when the
/// command succeeded).
pub fn text_output(bytes: &[u8]) -> std::result::Result<String, String> {
    pyjson::decode_utf8(bytes).map(universal_newlines)
}

/// `subprocess.run(..., capture_output=True, text=True)`: stdout and stderr both decoded as
/// [`text_output`] (stdout first, as `communicate` decodes them), or the stream that could not be
/// decoded and CPython's message.
pub fn text_streams(
    output: &std::process::Output,
) -> std::result::Result<(String, String), String> {
    let stdout = text_output(&output.stdout).map_err(|e| format!("stdout: {e}"))?;
    let stderr = text_output(&output.stderr).map_err(|e| format!("stderr: {e}"))?;
    Ok((stdout, stderr))
}

/// What Python 3.12's pathlib predicates (`exists`, `is_dir`, `is_file`, `is_symlink`) answered for
/// a path they could not inspect: `False` for a path that is missing (`pathlib._ignore_error`:
/// `ENOENT`, `ENOTDIR`, `EBADF`, `ELOOP`) or holds a NUL byte (a `ValueError` there), and the
/// `OSError` raised for anything else (a permission refused, a name too long), which is returned
/// here for the caller to refuse. No other error is ever read as "absent".
fn inspected(
    result: std::io::Result<std::fs::Metadata>,
) -> std::io::Result<Option<std::fs::Metadata>> {
    const ELOOP: i32 = if cfg!(target_os = "macos") { 62 } else { 40 };
    const IGNORED: [i32; 4] = [2, 9, 20, ELOOP]; // ENOENT, EBADF, ENOTDIR, ELOOP
    match result {
        Ok(meta) => Ok(Some(meta)),
        Err(e) if e.raw_os_error().is_some_and(|code| IGNORED.contains(&code)) => Ok(None),
        Err(e) if e.raw_os_error().is_none() && e.kind() == std::io::ErrorKind::InvalidInput => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// `Path.exists()` (Python 3.12): `False` for a path that is missing (`ENOENT`, `ENOTDIR`,
/// `EBADF`, `ELOOP`) or holds a NUL byte (a `ValueError` there); any other `OSError` (a name too
/// long, a permission refused) was raised, and is returned here.
pub fn path_exists(path: &Path) -> std::io::Result<bool> {
    Ok(inspected(std::fs::metadata(path))?.is_some())
}

/// `Path.is_dir()` (Python 3.12, following symlinks), failing as [`path_exists`] fails.
pub fn path_is_dir(path: &Path) -> std::io::Result<bool> {
    Ok(inspected(std::fs::metadata(path))?.is_some_and(|m| m.is_dir()))
}

/// `Path.is_file()` (Python 3.12, following symlinks), failing as [`path_exists`] fails.
pub fn path_is_file(path: &Path) -> std::io::Result<bool> {
    Ok(inspected(std::fs::metadata(path))?.is_some_and(|m| m.is_file()))
}

/// `Path.is_symlink()` (Python 3.12, `lstat`), failing as [`path_exists`] fails.
pub fn path_is_symlink(path: &Path) -> std::io::Result<bool> {
    Ok(inspected(std::fs::symlink_metadata(path))?.is_some_and(|m| m.file_type().is_symlink()))
}

/// `str(OSError)` for an error Python raised without a file name (`os.getcwd()`, say):
/// `[Errno N] <strerror>`.
pub fn os_error_bare(error: &std::io::Error) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            format!("[Errno {code}] {strerror}")
        }
        None => error.to_string(),
    }
}

/// `CompletedProcess.returncode`: the exit code, or minus the signal that killed the process.
pub fn returncode(status: std::process::ExitStatus) -> i64 {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => i64::from(code),
        (None, Some(signal)) => -i64::from(signal),
        (None, None) => -1,
    }
}

/// `str(subprocess.CalledProcessError(returncode, argv))` for `command`, the `repr` of the argument
/// list: `Command '[...]' returned non-zero exit status N.`, or for a command killed by a signal
/// `died with <Signals.SIGKILL: 9>.` (Linux signal numbers; an alias prints under its enum name,
/// `SIGABRT`, `SIGCHLD`, `SIGIO`).
pub fn called_process_error(command: &str, returncode: i64) -> String {
    const SIGNALS: [&str; 31] = [
        "SIGHUP",
        "SIGINT",
        "SIGQUIT",
        "SIGILL",
        "SIGTRAP",
        "SIGABRT",
        "SIGBUS",
        "SIGFPE",
        "SIGKILL",
        "SIGUSR1",
        "SIGSEGV",
        "SIGUSR2",
        "SIGPIPE",
        "SIGALRM",
        "SIGTERM",
        "SIGSTKFLT",
        "SIGCHLD",
        "SIGCONT",
        "SIGSTOP",
        "SIGTSTP",
        "SIGTTIN",
        "SIGTTOU",
        "SIGURG",
        "SIGXCPU",
        "SIGXFSZ",
        "SIGVTALRM",
        "SIGPROF",
        "SIGWINCH",
        "SIGIO",
        "SIGPWR",
        "SIGSYS",
    ];
    if returncode >= 0 {
        return format!("Command '{command}' returned non-zero exit status {returncode}.");
    }
    let signal = -returncode;
    let name = match signal {
        1..=31 => Some(SIGNALS[(signal - 1) as usize]),
        34 => Some("SIGRTMIN"),
        64 => Some("SIGRTMAX"),
        _ => None,
    };
    match name {
        Some(name) => format!("Command '{command}' died with <Signals.{name}: {signal}>."),
        None => format!("Command '{command}' died with unknown signal {signal}."),
    }
}

/// `str(OSError)` for an error Python raised with a file name: `[Errno N] <strerror>: '<name>'`
/// (`strerror` is the C library's text, which Rust's `io::Error` carries before `(os error N)`).
pub fn os_error(error: &std::io::Error, filename: &str) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            format!("[Errno {code}] {strerror}: {}", pyjson::str_repr(filename))
        }
        None => format!("{filename}: {error}"),
    }
}

// Names that are not UTF-8. Python held a file name, an argument or an environment value it could
// not decode as a `str` with each undecodable byte a lone surrogate (`os.fsdecode`,
// `surrogateescape`), and wrote that `str` back as the same bytes only through a stream or API
// with the `surrogateescape` handler. The ports never convert such a name lossily (U+FFFD): a
// value that is printed, written, compared or recorded refuses instead, and a message names it as
// Python printed it (`repr`, or `\udcXX` for a surrogate).

/// `os.fsdecode(name)`: the name's characters, each byte of an undecodable sequence a lone
/// surrogate `0xdc00 | byte` (`Err`), as Python's `surrogateescape` decoded it (one surrogate per
/// byte of each maximal invalid subpart).
pub fn fsdecode(name: &std::ffi::OsStr) -> Vec<std::result::Result<char, u16>> {
    use std::os::unix::ffi::OsStrExt;
    let mut units = vec![];
    let mut rest = name.as_bytes();
    loop {
        match std::str::from_utf8(rest) {
            Ok(text) => {
                units.extend(text.chars().map(Ok));
                return units;
            }
            Err(error) => {
                let (good, bad) = rest.split_at(error.valid_up_to());
                units.extend(
                    std::str::from_utf8(good)
                        .expect("the valid prefix")
                        .chars()
                        .map(Ok),
                );
                let n = error.error_len().unwrap_or(bad.len());
                units.extend(bad[..n].iter().map(|&b| Err(0xdc00 | u16::from(b))));
                rest = &bad[n..];
            }
        }
    }
}

/// `repr(os.fsdecode(name))`: an undecodable byte prints as `\udcXX` (`'/r/\udcff'`).
pub fn os_repr(name: &std::ffi::OsStr) -> String {
    pyjson::units_repr(&fsdecode(name))
}

/// [`os_error`] for a path given as the bytes it is: `[Errno N] <strerror>: '<repr>'`, the name
/// as `repr` printed Python's surrogate-escaped `str` (an error without an OS code: the name as
/// [`os_display`] renders it, then the error).
pub fn os_error_path(error: &std::io::Error, path: &Path) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            format!("[Errno {code}] {strerror}: {}", os_repr(path.as_os_str()))
        }
        None => format!("{}: {error}", os_display(path.as_os_str())),
    }
}

/// `str(os.fsdecode(name))` as Python wrote it to stderr, whose error handler is
/// `backslashreplace`: the name, each undecodable byte as the six characters `\udcXX`. For a
/// message only; a name that is not UTF-8 is never text a port prints, writes or compares.
pub fn os_display(name: &std::ffi::OsStr) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for unit in fsdecode(name) {
        match unit {
            Ok(c) => out.push(c),
            Err(surrogate) => {
                let _ = write!(out, "\\u{surrogate:04x}");
            }
        }
    }
    out
}

/// The `UnicodeEncodeError` CPython 3.12 raised writing `os.fsdecode(name)` through the strict
/// `utf-8` codec (stdout under a UTF-8 locale other than C or POSIX): `'utf-8' codec can't encode
/// character '\udcff' in position 5: surrogates not allowed`, the position counted in characters
/// and a run of surrogates one range (`characters in position 5-6`), as the codec reported the
/// first run. `None` for a name that is UTF-8.
pub fn utf8_encode_error(name: &std::ffi::OsStr) -> Option<String> {
    let units = fsdecode(name);
    let start = units.iter().position(std::result::Result::is_err)?;
    let run = units[start..].iter().take_while(|u| u.is_err()).count();
    Some(match (run, units[start]) {
        (1, Err(surrogate)) => format!(
            "'utf-8' codec can't encode character '\\u{surrogate:04x}' in position {start}: surrogates not allowed"
        ),
        _ => format!(
            "'utf-8' codec can't encode characters in position {start}-{}: surrogates not allowed",
            start + run - 1
        ),
    })
}

/// A pattern written for Python 3.12's `re` (a `str` pattern, Unicode 15.0), compiled so that its
/// classes match what Python's matched. The `regex` crate's own `\s`, `\w` and `\d` are Unicode
/// 16.0 `White_Space`, `Alphabetic`+marks+connectors and `Nd`; Python's are `str.isspace`
/// (`White_Space` plus U+001C..U+001F), `str.isalnum` or `_` (general category `L*` or `N*`)
/// and `str.isdecimal` (`Nd`), all as of Unicode 15.0. Those three escapes are rewritten, inside
/// or outside a bracket class; `\S`, `\W`, `\D`, `\b` and `\B` are not supported (none of the
/// ported patterns uses them).
///
/// # Panics
///
/// Panics if `pattern` ends with an incomplete escape, uses one of the unsupported escapes, or
/// is not a valid regex after translation. This helper is intended for the fixed verifier patterns
/// in this crate, not untrusted or arbitrary caller-provided patterns.
pub fn regex(pattern: &str) -> regex::Regex {
    const SPACE: &str = r"\s\x1C-\x1F";
    const WORD: &str = r"[\p{L}\p{N}_&&\p{Age=V15_0}]";
    const DIGIT: &str = r"[\p{Nd}&&\p{Age=V15_0}]";
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut depth = 0usize;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let escaped = chars.next().expect("a complete escape");
                match (escaped, depth > 0) {
                    ('s', false) => {
                        out.push('[');
                        out.push_str(SPACE);
                        out.push(']');
                    }
                    ('s', true) => out.push_str(SPACE),
                    ('w', _) => out.push_str(WORD),
                    ('d', _) => out.push_str(DIGIT),
                    ('S' | 'W' | 'D' | 'b' | 'B', _) => {
                        panic!("\\{escaped} has no Python-compatible translation here")
                    }
                    (other, _) => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            }
            '[' => {
                depth += 1;
                out.push(c);
                // A `]` or `^]` right after the opening bracket is literal in both dialects.
            }
            ']' if depth > 0 => {
                depth -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    regex::Regex::new(&out).expect("translated pattern")
}
