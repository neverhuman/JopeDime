//! Small Python built-in semantics the verifiers depend on: `str.strip`, hashing equality for set
//! membership, `in`, iteration, `Path.read_text`, `PurePosixPath.parts`, `os.path.realpath` and
//! `Path.resolve()`, the `re` classes of a `str` pattern, subprocess output read with `text=True`,
//! and the text of the `OSError` and `CalledProcessError` a tool printed.
//!
//! Numbers are the exception: text is read as a number by one strict grammar ([`decimal`],
//! [`int`], [`decimal_int`], [`float`]) that accepts a subset of what Python's `int()` and
//! `float()` accepted, never more (see [`decimal`]).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::pyjson::{self, Value};
use crate::{Error, Result};

/// Python `str.isspace` for one character (Unicode White_Space plus U+001C..U+001F).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// CPython 3.12's default `sys.get_int_max_str_digits()`. `int()` of a decimal string, and so
/// `json.loads` and `tomllib` of an integer literal, raised `ValueError` for more digits than this
/// (the sign, surrounding whitespace and underscores are not digits; leading zeros are).
pub const INT_MAX_STR_DIGITS: usize = 4300;

/// The limit on decimal integer conversion: `digits` digits are converted, more than
/// [`INT_MAX_STR_DIGITS`] refuse with CPython's `ValueError` message.
pub fn check_int_digits(digits: usize) -> std::result::Result<(), String> {
    if digits > INT_MAX_STR_DIGITS {
        return Err(format!(
            "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ));
    }
    Ok(())
}

/// Why [`decimal`] or [`int`] refused its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntError {
    /// Not a decimal integer of the strict grammar ([`decimal`]), or for [`int`] a value beyond
    /// `i64`. Each caller words this refusal.
    Invalid,
    /// A decimal integer of more than [`INT_MAX_STR_DIGITS`] digits: CPython's `ValueError`
    /// message ([`check_int_digits`]).
    Limit(String),
}

/// The whitespace the strict number grammar allows around a number: ASCII space, `\t`, `\n`,
/// `\r`, `\x0b` and `\x0c`, CPython's `Py_ISSPACE`. `int()` and `float()` of a `str` also stripped
/// the Unicode whitespace at or above U+0080 (U+0085, U+00A0, U+2003, ...), which the grammar
/// refuses; they refused U+001C..U+001F, which `str.strip()` removes, and so does the grammar.
pub fn is_number_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}')
}

/// A decimal integer read by the strict grammar ([`decimal`]): its sign and its ASCII digits as
/// written (leading zeros kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decimal<'a> {
    pub negative: bool,
    pub digits: &'a str,
}

impl Decimal<'_> {
    /// The digits without leading zeros (`0` for zero): `str(abs(int(text)))`, of any length.
    pub fn magnitude(&self) -> &str {
        let trimmed = self.digits.trim_start_matches('0');
        if trimmed.is_empty() {
            "0"
        } else {
            trimmed
        }
    }

    /// The value, when it fits `i64`.
    pub fn to_i64(&self) -> Option<i64> {
        let magnitude: i128 = self.magnitude().parse().ok()?;
        i64::try_from(if self.negative { -magnitude } else { magnitude }).ok()
    }
}

/// The one grammar every port reads a decimal integer from text with. It accepts a subset of what
/// Python's `int(text)` accepted, never more: optional surrounding whitespace of
/// [`is_number_space`] only, an optional ASCII sign (`+` or `-`), then one or more ASCII digits,
/// at most [`INT_MAX_STR_DIGITS`] of them (leading zeros count, as they did for CPython 3.12,
/// which refused more with the message [`check_int_digits`] words). Python also read single
/// underscores between digits (PEP 515), every other Unicode decimal digit (mapped to ASCII) and
/// Unicode whitespace at or above U+0080; the grammar refuses all three, and each port documents
/// that as stricter than the Python it replaced. The text is validated before its digits are
/// counted, as CPython validated a literal before it applied the limit.
pub fn decimal(text: &str) -> std::result::Result<Decimal<'_>, IntError> {
    let body = text.trim_matches(is_number_space);
    let (negative, digits) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(IntError::Invalid);
    }
    check_int_digits(digits.len()).map_err(IntError::Limit)?;
    Ok(Decimal { negative, digits })
}

/// `int(text)` through the strict grammar ([`decimal`]): the value, or [`IntError::Invalid`] for
/// text outside the grammar and for a value beyond `i64` (which Python converted; stricter).
pub fn int(text: &str) -> std::result::Result<i64, IntError> {
    decimal(text)?.to_i64().ok_or(IntError::Invalid)
}

/// A hashable JSON scalar under Python's hash equality (`1 == 1.0 == True`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    None,
    Int(i64),
    /// An integer beyond `i64` (a [`Value::BigInt`], or a float equal to one), as its decimal.
    Big(String),
    Float(u64),
    Str(String),
}

impl Key {
    /// `None` for an unhashable value (a list or a dict).
    pub fn of(value: &Value) -> Option<Key> {
        Some(match value {
            Value::Null => Key::None,
            Value::Bool(b) => Key::Int(i64::from(*b)),
            Value::Int(i) => Key::Int(*i),
            Value::BigInt(i) => Key::Big(i.clone()),
            Value::Float(f) => {
                if value == &Value::Int(*f as i64) {
                    Key::Int(*f as i64)
                } else if let Some(integer) = pyjson::integral_float_text(*f) {
                    Key::Big(integer)
                } else {
                    Key::Float(f.to_bits())
                }
            }
            Value::Str(s) => Key::Str(s.clone()),
            Value::List(_) | Value::Dict(_) => return None,
        })
    }

    pub fn str(s: &str) -> Key {
        Key::Str(s.to_owned())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Key::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Python `iter(value)` as a list of values: a list's items, a dict's keys, a string's
/// characters. `None` when Python would raise `TypeError`.
pub fn iterate(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::List(list) => Some(list.clone()),
        Value::Dict(dict) => Some(dict.keys().map(Value::from).collect()),
        Value::Str(s) => Some(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        _ => None,
    }
}

/// `set(value)` as hash keys; `None` when not iterable or an item is unhashable.
pub fn key_set(value: &Value) -> Option<std::collections::HashSet<Key>> {
    iterate(value)?.iter().map(Key::of).collect()
}

/// Python `item in container`; `None` when Python would raise `TypeError`. A list is searched
/// with identity first ([`pyjson::eq_or_identical`]), so a JSON `NaN` is in a list holding one.
pub fn contains(container: &Value, item: &Value) -> Option<bool> {
    match container {
        Value::List(list) => Some(list.iter().any(|x| pyjson::eq_or_identical(x, item))),
        Value::Dict(dict) => {
            let key = Key::of(item)?;
            Some(key.as_str().is_some_and(|k| dict.contains_key(k)))
        }
        Value::Str(s) => item.as_str().map(|needle| s.contains(needle)),
        _ => None,
    }
}

/// Python `item in container`, or the refusal for the pair Python could not search (see
/// [`exception::membership`]): which of the two values was the wrong type, and what it was.
pub fn contains_or_raise(container: &Value, item: &Value) -> std::result::Result<bool, String> {
    contains(container, item).ok_or_else(|| exception::membership(container, item))
}

/// `type(value).__name__` for a JSON value.
pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::BigInt(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
    }
}

/// Refusals for the malformed values a retired Python tool crashed on (a traceback and exit
/// status 1, no report). A port that meets such a value refuses instead of skipping it or carrying
/// on with a default. The caller prefixes the field; each message names the operation that could
/// not be applied and the offending value's type. The wording is modelled on the exception
/// CPython raised so a reader can recognise the case, but it is not a copy of CPython's text and
/// no test compares it with CPython's: the negative parity fixtures assert only that both sides
/// refuse (exit status 1, nothing printed on stdout).
pub mod exception {
    use super::type_name;
    use crate::pyjson::{str_repr, Value};

    /// `item in container` where Python could not search `container` for `item`: a dict and an
    /// unhashable item, a string and an item that is not a string, or a container that is neither
    /// a list, a dict nor a string.
    pub fn membership(container: &Value, item: &Value) -> String {
        match container {
            Value::Dict(_) => unhashable(item),
            Value::Str(_) => format!(
                "TypeError: 'in <string>' requires string as left operand, not {}",
                type_name(item)
            ),
            other => not_container(other),
        }
    }

    /// `Path(value)` on a value that is not a string.
    pub fn path_argument(value: &Value) -> String {
        format!(
            "TypeError: a path must be a string, not '{}'",
            type_name(value)
        )
    }

    /// `os.path.*(value)` (`os.fspath`) on a value that is not a string.
    pub fn fspath(value: &Value) -> String {
        format!(
            "TypeError: expected str, bytes or os.PathLike object, not {}",
            type_name(value)
        )
    }

    /// `for x in value`, `set(value)` or `s.intersection(value)` on a value that is not iterable.
    pub fn not_iterable(value: &Value) -> String {
        format!("TypeError: '{}' object is not iterable", type_name(value))
    }

    /// `item in value` on a value that is not a container.
    pub fn not_container(value: &Value) -> String {
        format!(
            "TypeError: argument of type '{}' is not iterable",
            type_name(value)
        )
    }

    /// Hashing a list or a dict (set membership, dict lookup).
    pub fn unhashable(value: &Value) -> String {
        format!("TypeError: unhashable type: '{}'", type_name(value))
    }

    /// `value.<attribute>` on a type that has no such attribute.
    pub fn no_attribute(value: &Value, attribute: &str) -> String {
        format!(
            "AttributeError: '{}' object has no attribute '{attribute}'",
            type_name(value)
        )
    }

    /// `d[key]` with the key missing.
    pub fn key_error(key: &str) -> String {
        format!("KeyError: {}", str_repr(key))
    }

    /// `value[key]` with a string key on a value that is not a dict.
    pub fn not_subscriptable(value: &Value) -> String {
        match value {
            Value::Str(_) => "TypeError: string indices must be integers, not 'str'".to_owned(),
            Value::List(_) => {
                "TypeError: list indices must be integers or slices, not str".to_owned()
            }
            other => format!(
                "TypeError: '{}' object is not subscriptable",
                type_name(other)
            ),
        }
    }
}

/// Python `separator.join(value)`: the items `for x in value` yields, joined, or the exception
/// line Python raised (a non-iterable, or an item that is not a string).
pub fn join(separator: &str, value: &Value) -> std::result::Result<String, String> {
    let items = iterate(value).ok_or_else(|| "TypeError: can only join an iterable".to_owned())?;
    let mut parts = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(text) => parts.push(text.to_owned()),
            None => {
                return Err(format!(
                    "TypeError: sequence item {i}: expected str instance, {} found",
                    type_name(item)
                ))
            }
        }
    }
    Ok(parts.join(separator))
}

/// Python `value[0]`: the first item (a list's, or a string's first character), or the exception
/// line Python raised.
pub fn first_item(value: &Value) -> std::result::Result<Value, String> {
    match value {
        Value::List(list) => list
            .first()
            .cloned()
            .ok_or_else(|| "IndexError: list index out of range".to_owned()),
        Value::Str(text) => text
            .chars()
            .next()
            .map(|c| Value::from(c.to_string()))
            .ok_or_else(|| "IndexError: string index out of range".to_owned()),
        Value::Dict(_) => Err("KeyError: 0".to_owned()),
        other => Err(format!(
            "TypeError: '{}' object is not subscriptable",
            type_name(other)
        )),
    }
}

/// Python `value[key]` for a string key: the item, or the refusal where Python raised (a missing
/// key, or a value that is not a dict).
pub fn subscript<'a>(value: &'a Value, key: &str) -> std::result::Result<&'a Value, String> {
    match value {
        Value::Dict(dict) => dict.get(key).ok_or_else(|| exception::key_error(key)),
        other => Err(exception::not_subscriptable(other)),
    }
}

/// Python `repr` of a tuple of values.
pub fn tuple_repr(items: &[Value]) -> String {
    let parts: Vec<String> = items.iter().map(Value::repr).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Python `repr` of a list of strings (as `sorted(...)[:n]` printed them).
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| pyjson::str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

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

/// `int(m.group(1))` of what a pattern's `\d+` captured, through the strict grammar ([`decimal`]):
/// the value as ASCII decimal without leading zeros, of any length up to [`INT_MAX_STR_DIGITS`]
/// digits, or the refusal. Python's `\d` matched every Unicode decimal digit and its `int()` mapped
/// them to ASCII; the grammar reads ASCII digits only and refuses a run holding any other
/// (stricter), naming it. More than 4300 digits refuse in CPython's words.
pub fn decimal_int(text: &str) -> std::result::Result<String, String> {
    match decimal(text) {
        Ok(Decimal {
            negative: false,
            digits,
        }) if digits.len() == text.len() => Ok(Decimal {
            negative: false,
            digits,
        }
        .magnitude()
        .to_owned()),
        Err(IntError::Limit(message)) => Err(message),
        _ => Err(format!(
            "{} is not ASCII decimal digits, the only digits the port reads as a number",
            pyjson::str_repr(text)
        )),
    }
}

/// Whether `text` is a run of what Python 3.12's `\d` matched (`str.isdecimal`, Unicode 15.0):
/// `re.fullmatch(r"\d+", text)`. A match only; a number is read from such a run by
/// [`decimal_int`].
pub fn is_decimal_run(text: &str) -> bool {
    static RUN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| regex(r"^\d+$"));
    RUN.is_match(text)
}

/// `json.loads(path.read_text())`.
pub fn load_json(path: &Path) -> Result<Value> {
    let text = read_text(path)?;
    pyjson::parse_str(&text).map_err(|message| Error::parse(path, message))
}

/// `json.loads(path.read_bytes())`.
pub fn load_json_bytes(path: &Path, bytes: &[u8]) -> Result<Value> {
    pyjson::parse_bytes(bytes).map_err(|message| Error::parse(path, message))
}

/// Lower-case hexadecimal SHA-256 (`hashlib.sha256(data).hexdigest()`).
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `Path.is_symlink()` read leniently: false when the path cannot be inspected. The P1 ports
/// (coverage, scope, parity, README, cargo-artifact, build-report) do not use it; they use
/// [`path_is_symlink`], which refuses instead. It remains for the record verifiers that predate
/// them.
pub fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// `os.path.abspath` (`posixpath`): joined to the working directory and normalised as
/// `posixpath.normpath` normalises, lexically: empty and `.` components dropped, `..` removing
/// the previous component (and nothing at the root), and exactly two leading slashes kept (POSIX
/// leaves `//` to the implementation, so Python keeps it as its own root; three or more are `/`).
/// Names stay the bytes they are. Lexical only: a symlink followed by `..` is [`realpath`]'s.
/// A relative path needs the working directory: when it cannot be read (it was deleted, say),
/// `os.getcwd()` raised and so does this, instead of resolving against `/`.
pub fn abspath(path: &Path) -> std::io::Result<PathBuf> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let mut joined = vec![];
    if !path.as_os_str().as_bytes().starts_with(b"/") {
        let cwd = std::env::current_dir()?;
        joined.extend_from_slice(cwd.as_os_str().as_bytes());
        if !joined.ends_with(b"/") {
            joined.push(b'/');
        }
    }
    joined.extend_from_slice(path.as_os_str().as_bytes());
    let slashes = if joined.starts_with(b"//") && !joined.starts_with(b"///") {
        2
    } else {
        1
    };
    let mut parts: Vec<&[u8]> = vec![];
    for part in joined.split(|&b| b == b'/') {
        match part {
            b"" | b"." => {}
            b".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    let mut out = b"/".repeat(slashes);
    out.extend(parts.join(&b'/'));
    Ok(PathBuf::from(std::ffi::OsString::from_vec(out)))
}

/// Why [`realpath`] or [`resolve`] could not resolve a path.
#[derive(Debug)]
pub enum ResolveError {
    /// `os.lstat` (strict mode) or `os.readlink` raised for `path`.
    Os {
        error: std::io::Error,
        path: PathBuf,
    },
    /// A symlink loop: pathlib raised `RuntimeError("Symlink loop from '<path>'")`.
    Loop { path: PathBuf },
    /// The working directory a relative path is resolved against could not be read:
    /// `os.getcwd()` raised (`[Errno 2] No such file or directory` for a deleted one).
    Cwd(std::io::Error),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Os { error, path } => f.write_str(&os_error_path(error, path)),
            ResolveError::Cwd(error) => f.write_str(&os_error_bare(error)),
            ResolveError::Loop { path } => {
                write!(f, "Symlink loop from {}", os_repr(path.as_os_str()))
            }
        }
    }
}

/// `os.path.realpath(path, strict=strict)`, CPython 3.12's `posixpath._joinrealpath` component by
/// component: each name is `lstat`ed and a symlink is replaced by its target, resolved in turn,
/// **before** a later `..` removes the last component of the path resolved so far, so
/// `outer/link/../x` with `link -> guarded/sub` is `guarded/x`, not the lexical `outer/x`.
/// Empty and `.` components are skipped and a leading `//` is `/` (the path is resolved from the
/// root). In strict mode a missing component (any `lstat` error) refuses naming the path up to it,
/// as Python's `OSError` did; otherwise the name is kept as it is spelled and resolution goes on.
/// `os.readlink` errors refuse in both modes. A symlink met again while it is being resolved is a
/// loop: in strict mode that refuses ([`ResolveError::Loop`], pathlib's `RuntimeError`); otherwise
/// the rest of the path is appended unresolved, as Python did. The result is [`abspath`] of the
/// resolved path (relative input is resolved against the working directory).
pub fn realpath(path: &Path, strict: bool) -> std::result::Result<PathBuf, ResolveError> {
    use std::collections::HashMap;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    fn os(bytes: &[u8]) -> &Path {
        Path::new(OsStr::from_bytes(bytes))
    }
    /// `posixpath.join(a, b)`.
    fn join(a: &[u8], b: &[u8]) -> Vec<u8> {
        if b.starts_with(b"/") {
            return b.to_vec();
        }
        let mut out = a.to_vec();
        if !(out.is_empty() || out.ends_with(b"/")) {
            out.push(b'/');
        }
        out.extend_from_slice(b);
        out
    }
    /// `posixpath.split(p)`.
    fn split(p: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let i = p.iter().rposition(|&b| b == b'/').map_or(0, |i| i + 1);
        let (head, tail) = p.split_at(i);
        let mut head = head.to_vec();
        if !head.is_empty() && head.iter().any(|&b| b != b'/') {
            while head.ends_with(b"/") {
                head.pop();
            }
        }
        (head, tail.to_vec())
    }
    type Seen = HashMap<Vec<u8>, Option<Vec<u8>>>;
    /// `_joinrealpath(path, rest, strict, seen)`: the resolved path and whether it is complete.
    fn join_realpath(
        mut path: Vec<u8>,
        rest: &[u8],
        strict: bool,
        seen: &mut Seen,
    ) -> std::result::Result<(Vec<u8>, bool), ResolveError> {
        let mut rest = rest;
        if rest.starts_with(b"/") {
            rest = &rest[1..];
            path = b"/".to_vec();
        }
        while !rest.is_empty() {
            let (name, tail) = match rest.iter().position(|&b| b == b'/') {
                Some(i) => (&rest[..i], &rest[i + 1..]),
                None => (rest, &b""[..]),
            };
            rest = tail;
            if name.is_empty() || name == b"." {
                continue;
            }
            if name == b".." {
                if path.is_empty() {
                    path = b"..".to_vec();
                } else {
                    let (head, last) = split(&path);
                    path = head;
                    if last == b".." {
                        path = join(&join(&path, b".."), b"..");
                    }
                }
                continue;
            }
            let newpath = join(&path, name);
            let is_link = match std::fs::symlink_metadata(os(&newpath)) {
                Ok(meta) => meta.file_type().is_symlink(),
                Err(error) if strict => {
                    return Err(ResolveError::Os {
                        error,
                        path: os(&newpath).to_path_buf(),
                    })
                }
                Err(_) => false,
            };
            if !is_link {
                path = newpath;
                continue;
            }
            match seen.get(&newpath) {
                Some(Some(resolved)) => {
                    path = resolved.clone();
                    continue;
                }
                Some(None) if strict => {
                    return Err(ResolveError::Loop {
                        path: os(&newpath).to_path_buf(),
                    })
                }
                Some(None) => return Ok((join(&newpath, rest), false)),
                None => {}
            }
            seen.insert(newpath.clone(), None);
            let target = std::fs::read_link(os(&newpath)).map_err(|error| ResolveError::Os {
                error,
                path: os(&newpath).to_path_buf(),
            })?;
            let (resolved, ok) = join_realpath(path, target.as_os_str().as_bytes(), strict, seen)?;
            path = resolved;
            if !ok {
                return Ok((join(&path, rest), false));
            }
            seen.insert(newpath, Some(path.clone()));
        }
        Ok((path, true))
    }
    let (resolved, _) = join_realpath(
        vec![],
        path.as_os_str().as_bytes(),
        strict,
        &mut HashMap::new(),
    )?;
    abspath(os(&resolved)).map_err(ResolveError::Cwd)
}

/// `Path(path).resolve(strict=strict)` (Python 3.12): [`realpath`], and in non-strict mode a
/// `stat` of the result, so a symlink loop refuses in both modes ([`ResolveError::Loop`], the
/// path `stat` named) while any other `stat` error is ignored, as pathlib ignored it. Every port
/// that resolves a path for a safety decision (custody, roots, "inside the tree") resolves it
/// here or with `std::fs::canonicalize`, which is the same resolution in strict mode; none
/// normalises `..` lexically before symlinks are resolved.
pub fn resolve(path: &Path, strict: bool) -> std::result::Result<PathBuf, ResolveError> {
    const ELOOP: i32 = if cfg!(target_os = "macos") { 62 } else { 40 };
    let resolved = realpath(path, strict)?;
    if !strict {
        if let Err(error) = std::fs::metadata(&resolved) {
            if error.raw_os_error() == Some(ELOOP) {
                return Err(ResolveError::Loop { path: resolved });
            }
        }
    }
    Ok(resolved)
}

/// `re.fullmatch("[0-9a-f]{n}", value or "")` for a JSON value (a non-string never matches).
pub fn is_hex(value: Option<&Value>, len: usize) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// `PurePosixPath(text)`: its root (`/`, or `//` exactly as POSIX keeps it) and its parts
/// without empty or `.` components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosixPath {
    pub root: Option<&'static str>,
    pub parts: Vec<String>,
}

impl PosixPath {
    pub fn parse(text: &str) -> Self {
        let root = if text.starts_with("//") && !text.starts_with("///") {
            Some("//")
        } else if text.starts_with('/') {
            Some("/")
        } else {
            None
        };
        let parts = text
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned)
            .collect();
        PosixPath { root, parts }
    }

    pub fn is_absolute(&self) -> bool {
        self.root.is_some()
    }

    /// `".." in path.parts`.
    pub fn has_parent_reference(&self) -> bool {
        self.parts.iter().any(|part| part == "..")
    }

    /// `base / path` for a relative path.
    pub fn join_onto(&self, base: &Path) -> PathBuf {
        let mut out = base.to_path_buf();
        for part in &self.parts {
            out.push(part);
        }
        out
    }

    /// Whether the relative parts begin with `prefix`.
    pub fn starts_with(&self, prefix: &[&str]) -> bool {
        self.root.is_none()
            && self.parts.len() >= prefix.len()
            && self.parts.iter().zip(prefix).all(|(a, b)| a == b)
    }

    /// `self / other`: an absolute `other` replaces the path, as in Python.
    pub fn join(&self, other: &str) -> Self {
        self.joinpath(&PosixPath::parse(other))
    }

    /// `self / other` for a parsed path: an absolute `other` (either root) replaces the path.
    pub fn joinpath(&self, other: &PosixPath) -> Self {
        if other.is_absolute() {
            return other.clone();
        }
        let mut parts = self.parts.clone();
        parts.extend(other.parts.iter().cloned());
        PosixPath {
            root: self.root,
            parts,
        }
    }

    /// `path.parent`: the path without its final part (a root, or `.`, is its own parent).
    pub fn parent(&self) -> Self {
        let mut parts = self.parts.clone();
        parts.pop();
        PosixPath {
            root: self.root,
            parts,
        }
    }

    /// `self.relative_to(other)` (Python 3.12, `walk_up=False`): the rest of the path when `other`
    /// is the path itself or one of its parents. pathlib compares them as spelled after parsing,
    /// so the roots must be the same (`/`, POSIX's distinct `//`, or none; `///` is `/`) and so
    /// must the leading parts. Otherwise the `ValueError` message, both paths as `str()` prints
    /// them.
    pub fn relative_to(&self, other: &PosixPath) -> std::result::Result<Self, String> {
        let n = other.parts.len();
        if self.root == other.root && self.parts.len() >= n && self.parts[..n] == other.parts[..] {
            return Ok(PosixPath {
                root: None,
                parts: self.parts[n..].to_vec(),
            });
        }
        Err(format!(
            "{} is not in the subpath of {}",
            pyjson::str_repr(&self.to_string()),
            pyjson::str_repr(&other.to_string())
        ))
    }

    /// `path.name`: the final component, or `""`.
    pub fn name(&self) -> &str {
        self.parts.last().map_or("", String::as_str)
    }

    /// `path.suffix` (Python 3.12): the final component's extension, if it has one that is not
    /// its whole name.
    pub fn suffix(&self) -> &str {
        let name = self.name();
        match name.rfind('.') {
            Some(i) if 0 < i && i < name.len() - 1 => &name[i..],
            _ => "",
        }
    }
}

/// `str(PurePosixPath(...))`.
impl std::fmt::Display for PosixPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self.parts.join("/");
        match (self.root, body.is_empty()) {
            (Some(root), _) => write!(f, "{root}{body}"),
            (None, true) => f.write_str("."),
            (None, false) => f.write_str(&body),
        }
    }
}

/// `math.fsum`: the correctly rounded sum of finite values, ported from CPython's
/// `mathmodule.c` (Shewchuk partials plus its half-even correction across partials). Returns
/// `None` where CPython raises (intermediate overflow, or `-inf + inf`).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0_f64;
    let mut inf_sum = 0.0_f64;
    for value in values {
        let mut x = value;
        let xsave = x;
        let mut kept = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if !x.is_finite() {
                if xsave.is_finite() {
                    return None; // intermediate overflow
                }
                if xsave.is_infinite() {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special_sum != 0.0 {
        return if inf_sum.is_nan() {
            None
        } else {
            Some(special_sum)
        };
    }
    let mut n = partials.len();
    let mut hi = 0.0_f64;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0_f64;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Some(hi)
}

/// Python `round(value, ndigits)` for a float: correctly rounded to `ndigits` decimals (ties to
/// even on the exact binary value), returned as the nearest double.
pub fn round(value: f64, ndigits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// `float(text)` through the strict number grammar: optional surrounding whitespace of
/// [`is_number_space`] only, an optional ASCII sign, then `digits [. [digits]]` or `. digits` in
/// ASCII digits, then optionally `e` or `E`, a sign and digits; the value is correctly rounded (an
/// exponent beyond the range of a double gives an infinity or zero, as Python's did). `None` for
/// anything else. Python also read `nan`, `inf` and `infinity` in any case, underscores between
/// digits, other Unicode decimal digits and Unicode whitespace at or above U+0080; no committed
/// input the ports read with `float()` writes any of them, and the grammar refuses them all
/// (stricter).
pub fn float(text: &str) -> Option<f64> {
    let body = text.trim_matches(is_number_space);
    let unsigned = body.strip_prefix(['+', '-']).unwrap_or(body);
    let s = unsigned.as_bytes();
    let digits = |mut i: usize| {
        while s.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        i
    };
    let mut i = digits(0);
    let mut any = i > 0;
    if s.get(i) == Some(&b'.') {
        let after = digits(i + 1);
        any |= after > i + 1;
        i = after;
    }
    if !any {
        return None;
    }
    if matches!(s.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(s.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let end = digits(j);
        if end == j {
            return None;
        }
        i = end;
    }
    if i != s.len() {
        return None;
    }
    body.parse().ok()
}

/// Python `str.splitlines()`: lines without their terminators.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = match c {
            '\r' => {
                if let Some(&(_, '\n')) = chars.peek() {
                    chars.next();
                    i + 2
                } else {
                    i + 1
                }
            }
            '\n' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        lines.push(&text[start..i]);
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_strip() {
        assert_eq!(strip("\u{1c} x \u{a0}"), "x");
        assert_eq!(strip("\u{1f}45"), "45");
    }

    /// The strict grammar reads what Python's `int()` read with ASCII whitespace, sign and digits.
    #[test]
    fn strict_int_reads_ascii_decimal() {
        assert_eq!(int("01"), Ok(1));
        assert_eq!(int("45"), Ok(45));
        assert_eq!(int(" \t\n\r\u{b}\u{c}+45 \t\n\r\u{b}\u{c}"), Ok(45));
        assert_eq!(int("-45"), Ok(-45));
        assert_eq!(int("-0"), Ok(0));
        assert_eq!(int(&i64::MAX.to_string()), Ok(i64::MAX));
        assert_eq!(int(&i64::MIN.to_string()), Ok(i64::MIN));
        let decimal = decimal(" -0045 ").unwrap();
        assert_eq!((decimal.negative, decimal.digits), (true, "0045"));
        assert_eq!(decimal.magnitude(), "45");
        assert_eq!(super::decimal("000").unwrap().magnitude(), "0");
    }

    /// Round four of the review of PR #117: every input Python's `int()` refused is refused, and
    /// so are the forms it read that the strict grammar does not (stricter): U+001C..U+001F
    /// (Python refused them though `str.strip()` removes them), Unicode whitespace at or above
    /// U+0080, underscores, other Unicode digits and values beyond `i64`.
    #[test]
    fn strict_int_refuses_what_python_refused_and_more() {
        for refused_by_python in [
            "",
            " ",
            "+",
            "-",
            "+-1",
            "1 1",
            "4.0",
            "0x10",
            "1e3",
            "45x",
            "\u{1c}45",
            "\u{1d}45",
            "\u{1e}45",
            "\u{1f}45",
            "45\u{1f}",
            "\u{200b}45",
            "1__0",
            "_1",
            "1_",
        ] {
            assert_eq!(
                int(refused_by_python),
                Err(IntError::Invalid),
                "{refused_by_python:?}"
            );
        }
        for read_by_python in [
            "\u{a0}45",
            "\u{85}45",
            "\u{2003}45",
            "\u{3000}45",
            "45\u{2028}",
            "4_5",
            "+1_0",
            "\u{664}\u{665}",
            "\u{ff14}\u{ff15}",
            "\u{1d7d2}",
            "9223372036854775808",
            "-9223372036854775809",
        ] {
            assert_eq!(
                int(read_by_python),
                Err(IntError::Invalid),
                "{read_by_python:?}"
            );
        }
    }

    #[test]
    fn pure_posix_path_parts() {
        let path = PosixPath::parse("evidence/./S03-G03//x.json");
        assert_eq!(path.parts, ["evidence", "S03-G03", "x.json"]);
        assert!(!path.is_absolute());
        assert!(PosixPath::parse("//a").root == Some("//"));
        assert!(PosixPath::parse("///a").root == Some("/"));
        assert!(PosixPath::parse("a/../b").has_parent_reference());
        assert!(PosixPath::parse("").parts.is_empty());
        let base = PosixPath::parse("assets/./track/");
        assert_eq!(
            base.join("tracks")
                .join("t1")
                .join("track.json")
                .to_string(),
            "assets/track/tracks/t1/track.json"
        );
        assert_eq!(base.join("/abs").to_string(), "/abs");
        assert_eq!(PosixPath::parse("").to_string(), ".");
        assert_eq!(PosixPath::parse("//x//y/").to_string(), "//x/y");
        assert_eq!(PosixPath::parse("a/scores.csv.zst").suffix(), ".zst");
        assert_eq!(PosixPath::parse("a/.zst").suffix(), "");
        assert_eq!(PosixPath::parse("a/b.").suffix(), "");
        assert_eq!(PosixPath::parse("a/b.json").name(), "b.json");
        assert_eq!(PosixPath::parse("a/..").name(), "..");
        assert_eq!(
            PosixPath::parse("./.github//workflows/ci.yml").parent(),
            PosixPath::parse(".github/workflows")
        );
        assert_ne!(
            PosixPath::parse("/.github/workflows/ci.yml").parent(),
            PosixPath::parse(".github/workflows")
        );
        assert_eq!(PosixPath::parse("/").parent().to_string(), "/");
        assert_eq!(PosixPath::parse("//a").parent().to_string(), "//");
        assert_eq!(PosixPath::parse("a").parent().to_string(), ".");
        assert_eq!(
            PosixPath::parse("h")
                .joinpath(&PosixPath::parse("//x"))
                .to_string(),
            "//x"
        );
    }

    /// `PurePosixPath.relative_to` (Python 3.12): exactly two leading slashes are a root of their
    /// own, in either direction and at the root itself; three or more are `/`. The refusal names
    /// both paths as `str()` prints them.
    #[test]
    fn relative_to_keeps_the_double_slash_root() {
        let rel = |a: &str, b: &str| {
            PosixPath::parse(a)
                .relative_to(&PosixPath::parse(b))
                .map(|p| p.to_string())
        };
        let outside = |a: &str, b: &str| Err(format!("'{a}' is not in the subpath of '{b}'"));
        assert_eq!(rel("/r/demo", "/r"), Ok("demo".to_owned()));
        assert_eq!(rel("//r/demo", "//r/"), Ok("demo".to_owned()));
        assert_eq!(rel("///r//demo", "/r"), Ok("demo".to_owned()));
        assert_eq!(rel("/r/demo", "///r"), Ok("demo".to_owned()));
        assert_eq!(rel("/r/demo", "/"), Ok("r/demo".to_owned()));
        assert_eq!(rel("//r/demo", "//"), Ok("r/demo".to_owned()));
        assert_eq!(rel("/r/demo", "/r/./demo/"), Ok(".".to_owned()));
        assert_eq!(rel("./bin/demo", "bin"), Ok("demo".to_owned()));
        assert_eq!(rel("a", ""), Ok("a".to_owned()));
        assert_eq!(rel("//r/demo", "/r"), outside("//r/demo", "/r"));
        assert_eq!(rel("/r/demo", "//r"), outside("/r/demo", "//r"));
        assert_eq!(rel("/r/demo", "//"), outside("/r/demo", "//"));
        assert_eq!(rel("//r/demo", "/"), outside("//r/demo", "/"));
        assert_eq!(rel("/rx/demo", "/r"), outside("/rx/demo", "/r"));
        assert_eq!(rel("/r", "/r/demo"), outside("/r", "/r/demo"));
        assert_eq!(rel("a/b", "/a"), outside("a/b", "/a"));
        assert_eq!(rel("/a/b", ""), outside("/a/b", "."));
        assert_eq!(rel("/a/../b", "/b"), outside("/a/../b", "/b"));
    }

    /// `os.path.realpath` and `Path.resolve()` resolve each symlink before a later `..` removes a
    /// component, as CPython 3.12's `posixpath._joinrealpath` did (round four of the review of
    /// PR #117: `outer/link/../receipts` with `link -> guarded/sub` is `guarded/receipts`).
    #[test]
    fn realpath_resolves_symlinks_before_parent_components() {
        let temp = tempfile::tempdir().unwrap();
        let c = temp.path().canonicalize().unwrap();
        for dir in ["outer/realdir", "guarded/sub"] {
            std::fs::create_dir_all(c.join(dir)).unwrap();
        }
        let link = |target: &Path, name: &str| std::os::unix::fs::symlink(target, c.join(name));
        link(&c.join("guarded/sub"), "outer/link").unwrap();
        link(Path::new("../guarded/sub"), "outer/rel").unwrap();
        link(Path::new("loop"), "outer/loop").unwrap();
        link(Path::new("nowhere"), "outer/dangling").unwrap();
        link(Path::new("b"), "outer/a").unwrap();
        link(Path::new("a"), "outer/b").unwrap();
        let at = |spelled: &str| PathBuf::from(format!("{}/{spelled}", c.display()));
        let lenient = |spelled: &str| resolve(&at(spelled), false).map_err(|e| e.to_string());
        for (spelled, resolved) in [
            ("outer/link/../receipts", "guarded/receipts"),
            ("outer/link/../sub/../receipts", "guarded/receipts"),
            ("outer/rel/../receipts", "guarded/receipts"),
            ("outer/missing/../link/../receipts", "guarded/receipts"),
            ("outer/link/..", "guarded"),
            ("outer/link/receipts", "guarded/sub/receipts"),
            ("outer/link/../../outer/receipts", "outer/receipts"),
            ("outer/realdir/../receipts", "outer/receipts"),
            ("outer/missing/../receipts", "outer/receipts"),
            ("outer/dangling/../receipts", "outer/receipts"),
            ("outer/dangling", "outer/nowhere"),
            ("outer/./link//receipts/", "guarded/sub/receipts"),
            // A loop's rest is appended unresolved and normalised, and `stat` of the result is
            // not a loop: resolved, as pathlib resolved it.
            ("outer/loop/../receipts", "outer/receipts"),
        ] {
            assert_eq!(lenient(spelled), Ok(c.join(resolved)), "{spelled}");
        }
        // POSIX's `//` root is the root `/` once resolved; `..` at the root stays there.
        let doubled = format!("/{}", at("outer/link/../receipts").display());
        assert_eq!(
            resolve(Path::new(&doubled), false).unwrap(),
            c.join("guarded/receipts")
        );
        assert_eq!(
            resolve(Path::new("/../.."), false).unwrap(),
            PathBuf::from("/")
        );
        // Symlink loops refuse (pathlib's `RuntimeError`), naming the path `stat` or `realpath` met.
        let spelled = at("outer/loop/x");
        assert_eq!(
            lenient("outer/loop/x"),
            Err(format!(
                "Symlink loop from {}",
                pyjson::str_repr(&spelled.to_string_lossy())
            ))
        );
        assert!(matches!(
            resolve(&at("outer/a"), false),
            Err(ResolveError::Loop { .. })
        ));
        assert!(matches!(
            resolve(&at("outer/loop"), true),
            Err(ResolveError::Loop { path }) if path == c.join("outer/loop")
        ));
        // Strict: a missing component refuses naming the path up to it.
        let missing = at("outer/missing");
        assert_eq!(
            resolve(&at("outer/missing/../receipts"), true).map_err(|e| e.to_string()),
            Err(format!(
                "[Errno 2] No such file or directory: {}",
                pyjson::str_repr(&missing.to_string_lossy())
            ))
        );
        assert_eq!(
            resolve(&at("outer/link/.."), true).unwrap(),
            c.join("guarded")
        );
        // Where everything exists, strict resolution is `realpath(3)`'s.
        for spelled in ["outer/link", "outer/rel/..", "outer/realdir/../link/."] {
            assert_eq!(
                resolve(&at(spelled), true).unwrap(),
                at(spelled).canonicalize().unwrap(),
                "{spelled}"
            );
        }
        // `abspath` alone is lexical.
        assert_eq!(
            abspath(&at("outer/link/../receipts")).unwrap(),
            c.join("outer/receipts")
        );
        assert_eq!(
            abspath(Path::new("//a/./b//c/")).unwrap().into_os_string(),
            "//a/b/c"
        );
        assert_eq!(
            abspath(Path::new("///a/../b")).unwrap().into_os_string(),
            "/b"
        );
    }

    /// The pathlib predicates answer "absent" only where Python 3.12 answered `False` (a missing
    /// path, a loop, a name through a file, a NUL byte); a permission refused or a name too long
    /// is an error, as Python raised it. No answer is read through a lenient default.
    #[test]
    fn predicates_refuse_what_pathlib_raised() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let c = temp.path().canonicalize().unwrap();
        std::fs::create_dir_all(c.join("dir/sub")).unwrap();
        std::fs::write(c.join("dir/file"), "").unwrap();
        std::os::unix::fs::symlink("loop", c.join("loop")).unwrap();
        std::os::unix::fs::symlink("nowhere", c.join("dangling")).unwrap();
        std::os::unix::fs::symlink(c.join("dir"), c.join("dirlink")).unwrap();
        assert!(path_is_dir(&c.join("dir")).unwrap());
        assert!(!path_is_dir(&c.join("dir/file")).unwrap());
        assert!(path_is_file(&c.join("dir/file")).unwrap());
        assert!(path_is_dir(&c.join("dirlink")).unwrap());
        assert!(path_is_symlink(&c.join("dirlink")).unwrap());
        assert!(!path_is_symlink(&c.join("dir")).unwrap());
        for absent in ["missing", "loop", "dangling", "dir/file/x", "nul\0x"] {
            let path = c.join(absent);
            assert!(!path_exists(&path).unwrap(), "{absent}");
            assert!(!path_is_dir(&path).unwrap(), "{absent}");
            assert!(!path_is_file(&path).unwrap(), "{absent}");
        }
        assert!(path_is_symlink(&c.join("loop")).unwrap());
        const ENAMETOOLONG: i32 = if cfg!(target_os = "macos") { 63 } else { 36 };
        let long = c.join("x".repeat(300));
        for probe in [path_exists, path_is_dir, path_is_file, path_is_symlink] {
            assert_eq!(probe(&long).unwrap_err().raw_os_error(), Some(ENAMETOOLONG));
        }
        // A directory that may be listed but not searched: its entries cannot be inspected.
        std::fs::set_permissions(c.join("dir"), std::fs::Permissions::from_mode(0o644)).unwrap();
        let denied = std::fs::metadata(c.join("dir/file")).is_err();
        if denied {
            for probe in [path_exists, path_is_dir, path_is_file, path_is_symlink] {
                assert_eq!(
                    probe(&c.join("dir/file")).unwrap_err().raw_os_error(),
                    Some(13)
                );
            }
        }
        std::fs::set_permissions(c.join("dir"), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            os_error_bare(&std::io::Error::from_raw_os_error(2)),
            "[Errno 2] No such file or directory"
        );
    }

    #[test]
    fn fsum_matches_cpython() {
        assert_eq!(fsum([0.1; 10]), Some(1.0));
        assert_eq!(fsum([1e-16, 1.0, 1e16]), Some(10000000000000002.0));
        assert_eq!(fsum([1e308, 1e308, -1e308]), None);
        assert_eq!(fsum([f64::INFINITY, 1.0]), Some(f64::INFINITY));
        assert_eq!(fsum([f64::INFINITY, f64::NEG_INFINITY]), None);
        assert_eq!(fsum([]), Some(0.0));
        assert_eq!(fsum([1.0, 1e100, 1.0, -1e100]), Some(2.0));
        // Plain left-to-right addition gives 0.9999999999999999 here.
        assert_ne!([0.1_f64; 10].iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn python_round_and_float() {
        assert_eq!(round(2.675, 2), 2.67);
        assert_eq!(round(0.25, 1), 0.2);
        assert_eq!((round(0.125, 2), round(0.375, 2)), (0.12, 0.38));
        assert_eq!(round(-0.04, 1).to_bits(), (-0.0_f64).to_bits());
    }

    /// `float()` through the strict grammar: the ASCII forms Python read, the same values.
    #[test]
    fn strict_float_reads_ascii_decimal_and_exponent_forms() {
        assert_eq!(float("21.511"), Some(21.511));
        assert_eq!(float(" \t\n\r\u{b}\u{c}21.511 \u{c}"), Some(21.511));
        assert_eq!(float("+4.622"), Some(4.622));
        assert_eq!(float("-4.622"), Some(-4.622));
        assert_eq!(float("1."), Some(1.0));
        assert_eq!(float(".5e-1"), Some(0.05));
        assert_eq!(float("4622E-3"), Some(4.622));
        assert_eq!(float("0.1e+01"), Some(1.0));
        assert_eq!(float("0021.5"), Some(21.5));
        assert_eq!(float("1e999"), Some(f64::INFINITY));
        assert_eq!(float("-1e999"), Some(f64::NEG_INFINITY));
        assert_eq!(float("1e-999"), Some(0.0));
        assert_eq!(float("2.675"), Some(2.675));
    }

    /// Round four of the review of PR #117: every input Python's `float()` refused is refused, and
    /// so are the forms it read that the strict grammar does not (stricter): `nan`, `inf` and
    /// `infinity`, underscores, other Unicode digits and Unicode whitespace at or above U+0080.
    #[test]
    fn strict_float_refuses_what_python_refused_and_more() {
        for refused_by_python in [
            "",
            " ",
            ".",
            "e5",
            "1e",
            "1e+",
            "1.5e",
            "0x10",
            "0x1p2",
            "1,5",
            "abc",
            "1 1",
            "\u{1c}21.5",
            "\u{1f}21.511",
            "21.511\u{1f}",
            "\u{200b}1",
            "1__0",
            "_1",
            "1_",
            "1.5_",
            "--1",
            "+-1",
            "infinit",
            "nana",
        ] {
            assert_eq!(float(refused_by_python), None, "{refused_by_python:?}");
        }
        for read_by_python in [
            "nan",
            "NaN",
            "-nan",
            "inf",
            "+inf",
            "-Infinity",
            "INFINITY",
            " 1_000.5 ",
            "1_0.5",
            "\u{a0}21.5",
            "\u{2003}21.5",
            "\u{85}21.5",
            "\u{664}.\u{666}",
            "\u{ff14}.5",
        ] {
            assert_eq!(float(read_by_python), None, "{read_by_python:?}");
        }
    }

    /// Python 3.12's `\s`, `\w` and `\d` (Unicode 15.0), not the `regex` crate's.
    #[test]
    fn regex_classes_are_pythons() {
        let space = regex(r"^\s$");
        for c in ['\u{1c}', '\u{1f}', '\u{a0}', '\u{2028}', '\u{3000}', '\t'] {
            assert!(space.is_match(&c.to_string()), "{c:?} is a Python space");
        }
        assert!(!space.is_match("\u{180e}") && !space.is_match("\u{200b}"));
        let word = regex(r"^\w$");
        for c in [
            '_', 'a', '\u{e9}', '\u{b2}', '\u{2167}', '\u{4e00}', '\u{661}',
        ] {
            assert!(
                word.is_match(&c.to_string()),
                "{c:?} is a Python word character"
            );
        }
        // Marks, connector punctuation, joiners and `Other_Alphabetic` symbols are not; nor is a
        // letter first assigned after Unicode 15.0 (U+2EBF0, CJK Extension I, 15.1).
        for c in ['\u{301}', '\u{203f}', '\u{200d}', '\u{24b6}', '\u{2ebf0}'] {
            assert!(
                !word.is_match(&c.to_string()),
                "{c:?} is not a Python word character"
            );
        }
        let class = regex(r"^[\w-]+\s*:$");
        assert!(class.is_match("a-b\u{1f}:") && !class.is_match("a\u{301}:"));
        let digit = regex(r"^\d+$");
        assert!(digit.is_match("0\u{665}\u{1d7ce}") && !digit.is_match("\u{b2}"));
        // What `\d+` matches is a match only; reading it as a number is the strict grammar's.
        assert!(is_decimal_run("0\u{665}\u{1d7d9}") && is_decimal_run("007"));
        assert!(!is_decimal_run("\u{b2}") && !is_decimal_run("") && !is_decimal_run("5\n"));
    }

    /// `int()` of a `\d+` capture through the strict grammar: ASCII digits of any value up to 4300
    /// digits (leading zeros count, as they did for CPython 3.12), 4301 refused in CPython's words;
    /// a run holding any other Unicode digit (which Python read) refused, naming it.
    #[test]
    fn int_conversion_limit_is_pythons() {
        let limit = |n: usize| {
            format!("Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit")
        };
        assert_eq!(check_int_digits(4300), Ok(()));
        assert_eq!(check_int_digits(4301), Err(limit(4301)));
        let run =
            |n: usize, zero: char, one: char| format!("{}{one}", zero.to_string().repeat(n - 1));
        assert_eq!(decimal_int(&run(4300, '0', '1')), Ok("1".to_owned()));
        assert_eq!(decimal_int(&run(4301, '0', '1')), Err(limit(4301)));
        assert_eq!(decimal_int(&"9".repeat(4300)), Ok("9".repeat(4300)));
        assert_eq!(decimal_int("000"), Ok("0".to_owned()));
        let not_ascii = |text: &str| {
            Err(format!(
                "{} is not ASCII decimal digits, the only digits the port reads as a number",
                pyjson::str_repr(text)
            ))
        };
        for read_by_python in [
            "\u{665}".to_owned(),
            "\u{ff10}\u{ff15}".to_owned(),
            "\u{1d7d3}".to_owned(),
            run(4300, '\u{660}', '\u{661}'),
            run(4302, '\u{660}', '\u{661}'),
        ] {
            assert_eq!(decimal_int(&read_by_python), not_ascii(&read_by_python));
        }
        for refused in ["\u{b2}", "", " 5", "+5", "-5", "5 "] {
            assert_eq!(decimal_int(refused), not_ascii(refused), "{refused:?}");
        }
        // `int` of text: beyond `i64` it refuses as invalid, past the limit in CPython's words.
        assert_eq!(int(&"1".repeat(20)), Err(IntError::Invalid));
        assert_eq!(int(&"1".repeat(4301)), Err(IntError::Limit(limit(4301))));
    }

    /// Python 3.12's `int()` of text: the limit is checked on a well-formed literal before it is
    /// converted, so 4301 digits refuse even when they are leading zeros of a small value; the
    /// sign and surrounding whitespace are not digits. The strict grammar refuses underscores
    /// (which Python read, not counting them as digits) whatever their number, and the reviewer's
    /// U+001F around the digits (which Python refused).
    #[test]
    fn int_of_text_counts_digits_as_python_did() {
        let limit = |n: usize| {
            IntError::Limit(format!("Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit"))
        };
        let zeros = |n: usize| "0".repeat(n);
        let unders = |n: usize| "0_".repeat(n);
        // 4300 digits convert, whatever ASCII whitespace and sign surround them.
        for text in [
            format!("{}45", zeros(4298)),
            format!("+{}45", zeros(4298)),
            format!(" \t{}45\n", zeros(4298)),
        ] {
            assert_eq!(int(&text), Ok(45), "{} characters", text.len());
        }
        assert_eq!(int(&format!("-{}45", zeros(4298))), Ok(-45));
        assert_eq!(int(&zeros(4300)), Ok(0));
        // 4301 refuse, leading zeros counted.
        for text in [
            format!("{}45", zeros(4299)),
            format!("+{}45", zeros(4299)),
            format!("-{}45", zeros(4299)),
            format!(" {}45 ", zeros(4299)),
            zeros(4301),
        ] {
            assert_eq!(int(&text), Err(limit(4301)), "{} characters", text.len());
        }
        assert_eq!(int(&zeros(5000)), Err(limit(5000)));
        // Outside the grammar, however many digits: underscores (Python read 4300 of them as 45
        // and refused 4301 at the limit), U+00A0 and U+001F around the digits.
        for text in [
            format!("{}45", unders(4298)),
            format!("{}45", unders(4299)),
            format!("\u{a0}+{}45\u{1f}", unders(4298)),
            format!("\u{1f}{}45", zeros(4298)),
        ] {
            assert_eq!(
                int(&text),
                Err(IntError::Invalid),
                "{} characters",
                text.len()
            );
        }
        // A malformed literal is refused as such, however long: CPython validated it first.
        assert_eq!(int(&format!("{}45_", zeros(4299))), Err(IntError::Invalid));
        assert_eq!(int(&format!("{}4x", zeros(4299))), Err(IntError::Invalid));
        // A non-ASCII digit stays refused (Python mapped it to ASCII and read 5; stricter).
        assert_eq!(
            int(&format!("{}\u{665}", zeros(4299))),
            Err(IntError::Invalid)
        );
    }

    #[test]
    fn subprocess_text_and_errors_read_as_python_printed_them() {
        assert_eq!(text_output(b"a\r\nb\rc\n").unwrap(), "a\nb\nc\n");
        assert_eq!(
            text_output(b"ok\xff").unwrap_err(),
            "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte"
        );
        assert_eq!(
            text_output(b"\xe2\x82x").unwrap_err(),
            "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"
        );
        assert_eq!(
            text_output(b"x\xed\xa0\x80").unwrap_err(),
            "'utf-8' codec can't decode byte 0xed in position 1: invalid continuation byte"
        );
        assert_eq!(
            text_output(b"x\xe2\x82").unwrap_err(),
            "'utf-8' codec can't decode bytes in position 1-2: unexpected end of data"
        );
        let missing = std::fs::metadata("/nonexistent-q/a").unwrap_err();
        assert_eq!(
            os_error(&missing, "/nonexistent-q/a"),
            "[Errno 2] No such file or directory: '/nonexistent-q/a'"
        );
        let argv = str_list_repr(&["du", "-s"]);
        assert_eq!(
            called_process_error(&argv, 1),
            "Command '['du', '-s']' returned non-zero exit status 1."
        );
        assert_eq!(
            called_process_error(&argv, -9),
            "Command '['du', '-s']' died with <Signals.SIGKILL: 9>."
        );
        assert_eq!(
            called_process_error(&argv, -40),
            "Command '['du', '-s']' died with unknown signal 40."
        );
        assert!(!path_exists(Path::new("/nonexistent-q/a")).unwrap());
        assert!(!path_exists(Path::new("/nonexistent-q/a\0b")).unwrap());
        const ENAMETOOLONG: i32 = if cfg!(target_os = "macos") { 63 } else { 36 };
        let long = format!("/{}", "x".repeat(300));
        assert_eq!(
            path_exists(Path::new(&long)).unwrap_err().raw_os_error(),
            Some(ENAMETOOLONG)
        );
    }

    /// Review round six of PR #117: a name that is not UTF-8 is rendered as Python rendered its
    /// surrogate-escaped `str` (never U+FFFD), and printing it through a strict UTF-8 stream fails
    /// with CPython's `UnicodeEncodeError` text: the first run of surrogates, positions in
    /// characters (`é` is one).
    #[test]
    fn names_that_are_not_utf8_render_as_python_rendered_them() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let name = |bytes: &[u8]| OsStr::from_bytes(bytes).to_owned();
        let ff = name(b"/r/\xff/demo");
        assert_eq!(os_repr(&ff), r"'/r/\udcff/demo'");
        assert_eq!(os_display(&ff), r"/r/\udcff/demo");
        let character = |at: usize, c: &str| {
            format!("'utf-8' codec can't encode character '\\{c}' in position {at}: surrogates not allowed")
        };
        let characters = |from: usize, to: usize| {
            format!("'utf-8' codec can't encode characters in position {from}-{to}: surrogates not allowed")
        };
        assert_eq!(utf8_encode_error(&ff), Some(character(3, "udcff")));
        // A run of undecodable bytes is one range; an incomplete sequence is one surrogate per byte.
        assert_eq!(
            utf8_encode_error(&name(b"/r/\xff\xfe/x")),
            Some(characters(3, 4))
        );
        assert_eq!(
            utf8_encode_error(&name(b"/r/\xe2\x82x/demo")),
            Some(characters(3, 4))
        );
        assert_eq!(
            fsdecode(&name(b"\xe2\x82x")),
            [Err(0xdce2), Err(0xdc82), Ok('x')]
        );
        // Positions count characters, and only the first run is reported.
        let mut accented = "/r/\u{e9}".as_bytes().to_vec();
        accented.extend_from_slice(b"\xff/demo");
        assert_eq!(
            utf8_encode_error(&name(&accented)),
            Some(character(4, "udcff"))
        );
        assert_eq!(
            utf8_encode_error(&name(b"/a\xffb\xfe")),
            Some(character(2, "udcff"))
        );
        assert_eq!(os_display(&name(b"/a\xffb\xfe")), r"/a\udcffb\udcfe");
        // A UTF-8 name, ASCII or not, is printed as it is.
        assert_eq!(utf8_encode_error(OsStr::new("/r/\u{e9}/demo")), None);
        assert_eq!(os_display(OsStr::new("/r/\u{e9}")), "/r/\u{e9}");
        assert_eq!(os_repr(OsStr::new("/r/it's")), "\"/r/it's\"");
    }

    #[test]
    fn python_splitlines() {
        assert_eq!(
            splitlines("a\nb\r\nc\rd\u{2028}e\n"),
            ["a", "b", "c", "d", "e"]
        );
        assert_eq!(splitlines("a\n\n"), ["a", ""]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn python_membership() {
        let list = pyjson::parse_str(r#"["a", 1]"#).unwrap();
        assert_eq!(contains(&list, &Value::Float(1.0)), Some(true));
        // Python `==`, not JSON-text equality: `True == 1 == 1.0`, `[1] == [1.0]`.
        assert_eq!(contains(&list, &Value::Bool(true)), Some(true));
        let nested = pyjson::parse_str("[[1]]").unwrap();
        assert_eq!(
            contains(&nested, &pyjson::parse_str("[1.0]").unwrap()),
            Some(true)
        );
        // Each refusal names the value of the wrong type.
        let dict = pyjson::parse_str(r#"{"a": 1}"#).unwrap();
        assert_eq!(
            contains_or_raise(&dict, &list),
            Err("TypeError: unhashable type: 'list'".to_owned())
        );
        assert_eq!(
            contains_or_raise(&Value::from("abc"), &Value::Int(1)),
            Err("TypeError: 'in <string>' requires string as left operand, not int".to_owned())
        );
        assert_eq!(
            contains_or_raise(&Value::Null, &Value::from("a")),
            Err("TypeError: argument of type 'NoneType' is not iterable".to_owned())
        );
        assert_eq!(contains_or_raise(&dict, &Value::Int(1)), Ok(false));
        assert_eq!(
            contains(&Value::from("abc"), &Value::from("bc")),
            Some(true)
        );
        assert_eq!(contains(&Value::Int(1), &Value::from("x")), None);
        assert_eq!(Key::of(&Value::Bool(true)), Key::of(&Value::Float(1.0)));
        // Every JSON NaN is one object: found by identity, though `NaN == NaN` is false.
        let nans = pyjson::parse_str("[NaN]").unwrap();
        assert_eq!(contains(&nans, &Value::Float(f64::NAN)), Some(true));
        assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
        assert_eq!(nans, pyjson::parse_str("[NaN]").unwrap());
        // Integers beyond 64 bits (opted in) compare and hash exactly, with floats too.
        let big = pyjson::parse_str_big("[18446744073709551616, 18446744073709551617]").unwrap();
        let items = big.as_list().unwrap();
        assert_ne!(items[0], items[1]);
        assert_eq!(items[0], Value::Float(18446744073709551616.0));
        assert_eq!(
            Key::of(&items[0]),
            Key::of(&Value::Float(18446744073709551616.0))
        );
        assert_eq!(
            contains(&big, &Value::Float(18446744073709551616.0)),
            Some(true)
        );
        assert_eq!(type_name(&items[1]), "int");
        assert!(pyjson::parse_str("18446744073709551616").is_err());
        assert_eq!(tuple_repr(&[Value::from("a"), Value::Null]), "('a', None)");
    }
}
